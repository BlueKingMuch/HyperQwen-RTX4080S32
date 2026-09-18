"""IQ3_XXS x bf16 GEMM in Gluon, the int8 form: the
IQ3_S int8 kernel with the IQ3_XXS block - 8-bit grid indices into the
256-word table (7-bit magnitudes 4..62), the four 7-bit sign indices of a
sub-block extended by their parity (popc) to 8-bit masks, the scale
d (2 s + 1) / 4. The B-fragment word b of a sub-block is grid word b; its
sign mask is nibble b mod 2 of the mask of group b // 2 (the four bits of
weights 4 (b mod 2) .. + 3), spread to byte masks and applied by the
byte-wise negation as in the IQ3_S form. Staging and the straddle fields:
the IQ3_XXS bf16 kernel (2-byte-aligned 98-byte blocks).
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

from .iq3s import split_k_for
from .iq3s_int8 import _field_in, _rowword_in, _word_to_i8x4, quantize_activations
from .iq3xxs import _popc, grid32

BLOCK_BYTES = 98


@g.jit
def iq3xxs_int8_kernel(XQ, SX, W, Y, GRID32,
                       M, N, num_k_blocks, row_bytes, stride_xq, stride_sx, stride_ym, stride_yk, w_shift, w_end,
                       BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 4], [32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 8])
    Lrow_w: gl.constexpr = SliceLayout(1, Lw)
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

    nrow_w = pid_n * BN + gl.arange(0, BN, layout=Lrow_w)
    drow_w = nrow_w.to(gl.int64) * row_bytes + w_shift
    nrow_c = pid_n * BN + gl.arange(0, BN, layout=Lrow_c)
    drow_c = nrow_c.to(gl.int64) * row_bytes + w_shift
    bidx = gl.arange(0, 8, layout=SliceLayout(0, Lw))
    zero2 = gl.zeros([64, 8], gl.int32, Lw)
    b2 = zero2 + bidx[None, :]
    gsh2 = 7 * (b2 >> 1)                        # the group's 7 sign bits in the u32
    nsh2 = 4 * (b2 & 1)                         # the word's nibble in the group's 8-bit mask
    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    o0 = crow + kb0 * 98
    src0 = (o0 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
    m0 = (cn < N)[:, None] & (cw < 25)[None, :] & (src0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem.index(0), (W + src0).to(gl.pointer_type(gl.int32), bitcast=True), m0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            o1 = crow + (kb + 1) * 98
            src1 = (o1 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
            m1 = (cn < N)[:, None] & (cw < 25)[None, :] & (src1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), (W + src1).to(gl.pointer_type(gl.int32), bitcast=True), m1)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        words = smem.index(it % 2)
        s2w = ((drow_w + kb * 98) & 2).to(gl.int32)
        selw = gl.where(s2w == 0, 0x5432, 0x7654)
        s2c = ((drow_c + kb * 98) & 2).to(gl.int32)
        selc = gl.where(s2c == 0, 0x5432, 0x7654)
        dw = _rowword_in(words, 0, Lcol_c, Lrow_c)
        d = ((dw >> (8 * s2c)) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            qsw0 = _field_in(words, 2 * ib, selw, Lcol_w, Lrow_w)               # grid indices of words 0..3
            qsw1 = _field_in(words, 2 * ib + 1, selw, Lcol_w, Lrow_w)           # words 4..7
            auxw = _field_in(words, 16 + ib, selw, Lcol_w, Lrow_w)              # signs (7 bits per group)
            auxc = _field_in(words, 16 + ib, selc, Lcol_c, Lrow_c)              # the same word for the scale, column layout
            qsw0_2 = zero2 + qsw0[:, None]
            qsw1_2 = zero2 + qsw1[:, None]
            aux_2 = zero2 + auxw[:, None]
            idx = gl.where(b2 < 4, qsw0_2 >> (8 * b2), qsw1_2 >> (8 * (b2 - 4))) & 0xFF
            gw = gl.load(GRID32 + idx)
            s7 = (aux_2 >> gsh2) & 127
            mask8 = s7 | ((_popc(s7) & 1) << 7)
            nib = (mask8 >> nsh2) & 0xF
            mask4 = ((nib * 0x204081) & 0x01010101) * 0xFF
            wq = (gw ^ mask4) + (mask4 & 0x01010101)
            w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
            b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
            acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
            s = (auxc >> 28) & 0xF
            dl = d * (2.0 * s.to(gl.float32) + 1.0) * 0.25
            acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def iq3xxs_int8_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                     quantized: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
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
    iq3xxs_int8_kernel[(triton.cdiv(n_out, BN), splitk)](
        XQ, SX, base, Y, grid32(W.device), M, n_out, nb, W.stride(0), XQ.stride(0), SX.stride(0), Y.stride(1), Y.stride(0), w_shift, w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
