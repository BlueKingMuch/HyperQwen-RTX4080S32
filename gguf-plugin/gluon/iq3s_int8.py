"""IQ3_S x bf16 GEMM in Gluon, the int8 form: the
lattice word decoded to four signed int8 weights in one register - the B
fragment of mma.m16n8k32.s8 - one mma per sub-block into an int32
accumulator, scaled into fp32 by the row's sub-block scale times the
token's activation scale.

Why: the bf16 forms are issue-bound at one byte per instruction (213
instructions per warp and sub-block, IPC 1.45, 300 GB/s). Here a thread
decodes two rows x two words per sub-block instead of eight bf16 pairs:
per word the index math as before, one gather, the sign nibble spread to
four byte masks ((nib * 0x204081) & 0x01010101) * 0xFF and the byte-wise
negation (w ^ mask) + (mask & 0x01010101), exact for magnitudes >= 1.

The activations are quantised ONCE per call by `quantize_activations`
(int8 per 32-wide group with an fp32 absmax scale - the plugin's MMVQ
quantises the same way, and its battery matched the bf16 decode's); the
GEMM loads its A fragments as int8 straight from that tensor and one scale
per row and sub-block. The first form quantised inside the sub-block loop,
replicated in every warp: about 250 instructions per thread and sub-block,
more than the decode it was meant to shrink.

Measured bases of the int8 operands on this image (gluon_int8_probe.py):
B [K=32, N=64] registers k+1, k+2, k+16, n+32; lanes k+4, k+8, n+1, n+2,
n+4; warps n+8, n+16. Staging, split-K, strided rows, the storage-end
mask: iq3s_gluon5.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, SliceLayout, DotOperandLayout, NVMMADistributedLayout, SwizzledSharedLayout,
    DistributedLinearLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from .iq3s import _prmt, grid32, split_k_for   # the bf16 kernel's table, prmt and split-K heuristic

BLOCK_BYTES = 110


@triton.jit
def _quantize_kernel(X, XQ, SX, SUMX, K, stride_x, stride_xq, stride_sx, GROUPS: tl.constexpr, WITH_SUMS: tl.constexpr):
    """One program per (row, span of GROUPS x 32 columns): every 32-wide group
    to int8 with its absmax / 127, and (WITH_SUMS) the group's sum of the
    quantised values times the scale - the term the K types' mins need."""
    m = tl.program_id(0)
    g0 = tl.program_id(1) * (32 * GROUPS)
    offs = g0 + tl.arange(0, 32 * GROUPS)
    x = tl.load(X + m * stride_x + offs, mask=offs < K, other=0.0).to(tl.float32)
    x2 = tl.reshape(x, [GROUPS, 32])
    amax = tl.max(tl.abs(x2), axis=1)
    sx = tl.where(amax > 0, amax / 127.0, 1.0)
    q = x2 / sx[:, None]
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)
    tl.store(XQ + m * stride_xq + offs, tl.reshape(q, [32 * GROUPS]).to(tl.int8), mask=offs < K)
    gi = g0 // 32 + tl.arange(0, GROUPS)
    tl.store(SX + m * stride_sx + gi, sx, mask=gi < K // 32)
    if WITH_SUMS:
        tl.store(SUMX + m * stride_sx + gi, tl.sum(q, axis=1).to(tl.float32) * sx, mask=gi < K // 32)


def quantize_activations(X: torch.Tensor, with_sums: bool = False):
    """X bf16 [M, K] -> (int8 [M, K], fp32 scales [M, K // 32]) and, with
    with_sums, the fp32 group sums [M, K // 32] of the dequantised values."""
    M, K = X.shape
    assert K % 32 == 0
    XQ = torch.empty((M, K), dtype=torch.int8, device=X.device)
    SX = torch.empty((M, K // 32), dtype=torch.float32, device=X.device)
    SUMX = torch.empty((M, K // 32), dtype=torch.float32, device=X.device) if with_sums else SX
    GROUPS = 8
    _quantize_kernel[(M, triton.cdiv(K, 32 * GROUPS))](X, XQ, SX, SUMX, K, X.stride(0), XQ.stride(0), SX.stride(0),
                                                        GROUPS=GROUPS, WITH_SUMS=with_sums, num_warps=2)
    return (XQ, SX, SUMX) if with_sums else (XQ, SX)


@g.jit
def _word_to_i8x4(w):
    """One int32 word -> four int8 elements (pack=4: $0 the register of the four
    outputs, $1..$4 the four copies of the word)."""
    return gl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r,r,r,r", [w], dtype=gl.int8, is_pure=True, pack=4)


@g.jit
def _rowword_in(wview, w: gl.constexpr, Lcol: gl.constexpr, Lrow: gl.constexpr):
    """Word column w of the [64, 32]-word tile as a [64] tensor in the 1-D row
    layout Lrow (loaded through its [64, 1] form Lcol)."""
    q = wview.slice(w, 1, dim=1).load(Lcol)
    return gl.convert_layout(gl.reshape(q, [64]), Lrow)


@g.jit
def _field_in(wview, w: gl.constexpr, sel, Lcol: gl.constexpr, Lrow: gl.constexpr):
    return _prmt(_rowword_in(wview, w, Lcol, Lrow), _rowword_in(wview, w + 1, Lcol, Lrow), sel)


@g.jit
def iq3s_int8_kernel(XQ, SX, W, Y, GRID32,
                     M, N, num_k_blocks, row_bytes, stride_xq, stride_sx, stride_ym, stride_yk, w_shift, w_end,
                     BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    # the decode layout over [64 rows, 8 words]: the B fragment's bases with the
    # four bytes of a word as one element (lanes k+4 -> word bit 0, k+8 -> bit
    # 1; register k+16 -> bit 2; rows: lanes 1, 2, 4, warps 8, 16, register 32)
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 4], [32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 8])
    Lrow_w: gl.constexpr = SliceLayout(1, Lw)
    Lcol_w: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]], lane_bases=[[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 1])
    # rows (n) as the accumulator's column slice (n+2, n+4 on lane bits 0, 1;
    # n+1, n+32 registers; n+8, n+16 warps; m on lane bits 2-4, broadcast)
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
    # A fragment indices (int8, in the dot-operand layout itself)
    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    # the activation scale per row and sub-block, in the accumulator's row slice
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
        qh01 = _field_in(words, 16, selw, Lcol_w, Lrow_w)
        qh23 = _field_in(words, 17, selw, Lcol_w, Lrow_w)
        s2c = ((drow_c + kb * 110) & 2).to(gl.int32)
        selc = gl.where(s2c == 0, 0x5432, 0x7654)
        dw = _rowword_in(words, 0, Lcol_c, Lrow_c)
        d = ((dw >> (8 * s2c)) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        scw = _field_in(words, 26, selc, Lcol_c, Lrow_c)
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)     # int8 [16, 32] in da8
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)                                  # [16] fp32
            qsw0 = _field_in(words, 2 * ib, selw, Lcol_w, Lrow_w)
            qsw1 = _field_in(words, 2 * ib + 1, selw, Lcol_w, Lrow_w)
            sgw = _field_in(words, 18 + ib, selw, Lcol_w, Lrow_w)
            if ib < 4:
                qh = (qh01 >> (8 * ib)) & 0xFF
            else:
                qh = (qh23 >> (8 * (ib - 4))) & 0xFF
            qsw0_2 = zero2 + qsw0[:, None]
            qsw1_2 = zero2 + qsw1[:, None]
            qh_2 = zero2 + qh[:, None]
            sg_2 = zero2 + sgw[:, None]
            qsb = gl.where(b2 < 4, qsw0_2 >> (8 * b2), qsw1_2 >> (8 * (b2 - 4))) & 0xFF
            idx = qsb | (((qh_2 >> b2) & 1) << 8)
            gw = gl.load(GRID32 + idx)                                           # [64, 8] magnitude words
            nib = (sg_2 >> (4 * b2)) & 0xF
            mask4 = ((nib * 0x204081) & 0x01010101) * 0xFF                      # a 0xFF byte per set sign bit
            wq = (gw ^ mask4) + (mask4 & 0x01010101)                             # four signed int8 weights
            w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))       # [64, 8, 2, 2] int8
            b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
            acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
            s = (scw >> (4 * ib)) & 0xF
            dl = d * (1.0 + 2.0 * s.to(gl.float32))                              # [64] per row (column slice)
            acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def iq3s_int8_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                   quantized: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """W: the raw IQ3_S rows (any row stride); X bf16 [M <= 16, K]; `quantized`
    = (int8 X, scales) from quantize_activations when the caller shares them
    across the layer's shards."""
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
    iq3s_int8_kernel[(triton.cdiv(n_out, BN), splitk)](
        XQ, SX, base, Y, grid32(W.device), M, n_out, nb, W.stride(0), XQ.stride(0), SX.stride(0), Y.stride(1), Y.stride(0), w_shift, w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
