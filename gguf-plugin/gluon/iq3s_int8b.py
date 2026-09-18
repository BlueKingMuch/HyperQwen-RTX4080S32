"""IQ3_S x bf16 GEMM in Gluon, the int8 form's third variant.

What changes per lattice word, all else as in the first int8 form:

  * the word index's bit 2 (word b vs b + 4: the register base k+16 of the
    fragment) is a Python-level split here - the eight words of a row's
    sub-block are two [64, 4] tensors, low and high, joined into the [64, 8]
    fragment tile at the end - so `where(b < 4, qs0, qs1)` and every
    `b - 4` disappear; the compiler saw those as runtime selects (ISETP +
    SEL per element) because b came from a runtime tensor;
  * the index byte comes out of its 32-bit field by one prmt against a zero
    register with a per-lane selector hoisted out of the loops (byte b of
    the field into byte 0, zeros elsewhere) instead of shift + mask;
  * the qh bit is shifted into bit 8 by one shift with a hoisted per-lane
    amount and or-ed by one lop3;
  * the byte-wise negation reuses the 0x01010101-masked spread as its +1
    term instead of masking again: (w ^ m) + x with m = x * 0xFF.

Per word: prmt, shift, lop3 (index); lea pair and the gather; shift and
lop3 (sign nibble); imad, lop3, imad (spread); lop3, iadd3 (negate) - 13
against about 30. MAXNREG (launcher argument) caps the registers to lift
the occupancy from 3 blocks per SM; measured, not assumed.
"""
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, SliceLayout, DotOperandLayout, NVMMADistributedLayout, SwizzledSharedLayout,
    DistributedLinearLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from .iq3s import _prmt, grid32, split_k_for
from .iq3s_int8 import _field_in, _rowword_in, _word_to_i8x4, quantize_activations

BLOCK_BYTES = 110


@g.jit
def _decode_half(qsw, qh, sgw, selq, shq, shs, zero4, GRID32):
    """[64, 4] words of one half of a sub-block (low: words 0..3, high: 4..7):
    the grid word of each, signs applied, as four int8 weights per word."""
    qs_4 = zero4 + qsw[:, None]
    qh_4 = zero4 + qh[:, None]
    sg_4 = zero4 + sgw[:, None]
    qsb = _prmt(qs_4, zero4, selq)                          # byte b of the field, zero-extended
    idx = qsb | ((qh_4 << shq) & 0x100)                     # the 9th bit from qh
    gw = gl.load(GRID32 + idx)                              # magnitude word
    nib = (sg_4 >> shs) & 0xF                               # the word's four sign bits
    x = (nib * 0x204081) & 0x01010101                       # a 0x01 byte per set sign bit
    return (gw ^ (x * 0xFF)) + x                            # byte-wise negation


@g.jit
def iq3s_int8b_kernel(XQ, SX, W, Y, GRID32,
                      M, N, num_k_blocks, row_bytes, stride_xq, stride_sx, stride_ym, stride_yk, w_shift, w_end,
                      BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    # the fragment's word tile without its register base k+16: one half of a
    # sub-block's eight words, [64 rows, 4 words] (lanes k+4, k+8 -> word bits
    # 0, 1; rows on lanes 1, 2, 4, warps 8, 16, register 32)
    L4: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 4])
    Lrow4: gl.constexpr = SliceLayout(1, L4)
    Lcol_w: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]], lane_bases=[[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 1])
    Lrow_c: gl.constexpr = SliceLayout(0, mma)
    Lcol_c: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [32, 0]], lane_bases=[[2, 0], [4, 0], [0, 0], [0, 0], [0, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 1])

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    cp: gl.constexpr = BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    cn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, cp))
    cw = gl.arange(0, 32, layout=SliceLayout(0, cp))
    crow = cn.to(gl.int64) * row_bytes + w_shift
    smem = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_words)

    nrow_w = pid_n * BN + gl.arange(0, BN, layout=Lrow4)
    drow_w = nrow_w.to(gl.int64) * row_bytes + w_shift
    nrow_c = pid_n * BN + gl.arange(0, BN, layout=Lrow_c)
    drow_c = nrow_c.to(gl.int64) * row_bytes + w_shift
    # per-lane constants of the word within the half (b = 0..3), hoisted
    bidx = gl.arange(0, 4, layout=SliceLayout(0, L4))
    zero4 = gl.zeros([64, 4], gl.int32, L4)
    b4 = zero4 + bidx[None, :]
    selq = b4 | 0x4440                                      # prmt: byte b of $1 -> byte 0, bytes 1..3 from $2 (zero)
    shq_lo = 8 - b4                                         # qh bit b -> bit 8 (low half: words 0..3)
    shq_hi = 4 - b4                                         # qh bit b + 4 -> bit 8 (high half)
    shs_lo = 4 * b4                                         # sign nibble of word b
    shs_hi = 16 + 4 * b4                                    # of word b + 4
    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    o0 = crow + kb0 * 110
    src0 = (o0 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
    m0 = (cn < N)[:, None] & (cw < 28)[None, :] & (src0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem.index(0), (W + src0).to(gl.pointer_type(gl.int32), bitcast=True), m0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            o1 = crow + (kb + 1) * 110
            src1 = (o1 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
            m1 = (cn < N)[:, None] & (cw < 28)[None, :] & (src1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), (W + src1).to(gl.pointer_type(gl.int32), bitcast=True), m1)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        words = smem.index(it % 2)
        s2w = ((drow_w + kb * 110) & 2).to(gl.int32)
        selw = gl.where(s2w == 0, 0x5432, 0x7654)
        qh01 = _field_in(words, 16, selw, Lcol_w, Lrow4)
        qh23 = _field_in(words, 17, selw, Lcol_w, Lrow4)
        s2c = ((drow_c + kb * 110) & 2).to(gl.int32)
        selc = gl.where(s2c == 0, 0x5432, 0x7654)
        dw = _rowword_in(words, 0, Lcol_c, Lrow_c)
        d = ((dw >> (8 * s2c)) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        scw = _field_in(words, 26, selc, Lcol_c, Lrow_c)
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            qsw0 = _field_in(words, 2 * ib, selw, Lcol_w, Lrow4)                 # index bytes of words 0..3
            qsw1 = _field_in(words, 2 * ib + 1, selw, Lcol_w, Lrow4)             # words 4..7
            sgw = _field_in(words, 18 + ib, selw, Lcol_w, Lrow4)                 # the sub-block's 32 sign bits
            if ib < 4:
                qh = (qh01 >> (8 * ib)) & 0xFF
            else:
                qh = (qh23 >> (8 * (ib - 4))) & 0xFF
            wq_lo = _decode_half(qsw0, qh, sgw, selq, shq_lo, shs_lo, zero4, GRID32)
            wq_hi = _decode_half(qsw1, qh, sgw, selq, shq_hi, shs_hi, zero4, GRID32)
            w8 = gl.join(wq_lo, wq_hi)                                            # [64, 4, 2]: (row, word b, half)
            w4 = _word_to_i8x4(gl.join(gl.join(w8, w8), gl.join(w8, w8)))        # [64, 4, 2, 2, 2] int8: (..., j1, j0)
            # k = 16 half + 4 b + 2 j1 + j0
            b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (2, 1, 3, 4, 0)), [32, 64]), db8)
            acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
            s = (scw >> (4 * ib)) & 0xF
            dl = d * (1.0 + 2.0 * s.to(gl.float32))
            acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def iq3s_int8b_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                    quantized: tuple | None = None, maxnreg: int | None = None) -> torch.Tensor:
    BN = 64
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * BLOCK_BYTES and M <= 16
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations(X.contiguous())
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=BLOCK_BYTES)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    off = W.storage_offset()
    base = torch.as_strided(W, (1,), (1,), off & -4)
    w_shift = off & 3
    w_end = W.untyped_storage().nbytes() - (off & -4)
    opts = {"maxnreg": maxnreg} if maxnreg else {}
    iq3s_int8b_kernel[(triton.cdiv(n_out, BN), splitk)](
        XQ, SX, base, Y, grid32(W.device), M, n_out, nb, W.stride(0), XQ.stride(0), SX.stride(0), Y.stride(1), Y.stride(0), w_shift, w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps, **opts)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
