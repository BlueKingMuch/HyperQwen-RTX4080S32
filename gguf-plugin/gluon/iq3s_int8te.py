"""The IQ3_S tile kernel (tile-major rows, 16-byte copies, the grid
table and the A tile in shared memory) with the sixth form's arithmetic:
the activation scale per 256 instead of
per 32, the sub-block scales applied as the integers they are, the eight
sub-blocks of a k-block accumulated in int32 (acc_k += C_i x (1 + 2 s_i),
one IMAD per element; |C_i| <= 32 x 127 x 15, x 31, x 8 is 15 M, far inside
int32) and one conversion per k-block, acc += float(acc_k) x sx[m] x d[n].
The per-32 fp32 epilogue of the tile kernel (8 cvt, 8 mul, 8 fma and the
scale arithmetic per thread and sub-block) becomes 8 IMAD per sub-block and
24 fp instructions per k-block. Everything else - the copies, the decode,
the mma - is the tile kernel's; against the row sixth form the result is
bit-identical (the same operations in the same order), against the per-32
forms it differs by the activations' rounding: the quality battery is its
gate, the executed instruction count and the in-situ time its measure.

    from .iq3s_int8te import iq3s_int8te_gemm
    y = iq3s_int8te_gemm(repack_iq3s_tiles(W, n_out), x, n_out, splitk=s)   # quantises x per 256 itself

Bit-identical to the row sixth form; the per-256 quantiser is inlined.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DistributedLinearLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from .splitk_reduce import _splitk_reduce, splitk_buffers, splitk_result   # 0039: the in-kernel split-K reduction (an option)

from .iq3s import _prmt, grid32, split_k_for
from .iq3s_int8 import _rowword_in, _word_to_i8x4
from .iq3s_int8t import TAB_OFF, _lds, _smem_base

BLOCK_BYTES = 110
TILE_ROW_BYTES = 112
TILE_BYTES = 64 * TILE_ROW_BYTES          # 7,168 = 448 x 16
TILE_WORDS = TILE_BYTES // 4


from .iq3s_int8t import repack_iq3s_tiles  # noqa: F401  (the same tiles)


@triton.jit
def _quantize256_kernel(X, XQ, SX, K, stride_x, stride_xq, stride_sx):
    """One program per (row, block of 256 columns): the block to int8 with its
    absmax / 127, the scale per block."""
    m = tl.program_id(0)
    b = tl.program_id(1)
    offs = b * 256 + tl.arange(0, 256)
    x = tl.load(X + m * stride_x + offs, mask=offs < K, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    sx = tl.where(amax > 0, amax / 127.0, 1.0)
    q = x / sx
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)
    tl.store(XQ + m * stride_xq + offs, q.to(tl.int8), mask=offs < K)
    tl.store(SX + m * stride_sx + b, sx)


def quantize_activations_256(X: torch.Tensor):
    """X bf16 [M, K] -> (int8 [M, K], fp32 scales [M, K // 256])."""
    M, K = X.shape
    assert K % 256 == 0
    XQ = torch.empty((M, K), dtype=torch.int8, device=X.device)
    SX = torch.empty((M, K // 256), dtype=torch.float32, device=X.device)
    _quantize256_kernel[(M, K // 256)](X, XQ, SX, K, X.stride(0), XQ.stride(0), SX.stride(0), num_warps=2)
    return XQ, SX


@g.jit
def _decode_half_t(qsw, qh, sgw, selq, shq, shs, zero4, tbase):
    qs_4 = zero4 + qsw[:, None]
    qh_4 = zero4 + qh[:, None]
    sg_4 = zero4 + sgw[:, None]
    qsb = _prmt(qs_4, zero4, selq)
    idx = qsb | ((qh_4 << shq) & 0x100)
    gw = _lds(tbase + idx * 4)
    nib = (sg_4 >> shs) & 0xF
    x = (nib * 0x204081) & 0x01010101
    return (gw ^ (x * 0xFF)) + x


@g.jit
def iq3s_int8te_kernel(XQ, SX, WT, Y, GRID32,
                      M, N, num_k_blocks, stride_xq, stride_sx, stride_ym, stride_yk, P, C, stride_pm, stride_pk,
                      BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr, REDUCE: gl.constexpr, TAB_OFF: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_words: gl.constexpr = SwizzledSharedLayout(4, 1, 8, [1, 0])     # 16-byte units, swizzled over 8 phases
    smem_a_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    smem_a8: gl.constexpr = SwizzledSharedLayout(4, 1, 8, [1, 0])
    tab_layout: gl.constexpr = SwizzledSharedLayout(1, 1, 1, [0])
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

    gtab = gl.allocate_shared_memory(gl.int32, [512], tab_layout)
    tl1: gl.constexpr = BlockedLayout([4], [32], [4], [0])
    ti = gl.arange(0, 512, layout=tl1)
    gtab.store(gl.load(GRID32 + ti))
    smem = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_words)
    smem_a = gl.allocate_shared_memory(gl.int32, [2, BM, 64], smem_a_words)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    # the tile copy: 64 rows x 28 words as 16-byte chunks, 8 lanes per row, rows 28..31 of the smem tile unwritten
    cp16: gl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    cr = gl.arange(0, 64, layout=SliceLayout(1, cp16))
    cc = gl.arange(0, 32, layout=SliceLayout(0, cp16))
    cc = gl.max_contiguous(gl.multiple_of(cc, 4), 4)
    cmask = (cc < 28)[None, :] & (cr < 64)[:, None]
    WT32 = WT.to(gl.pointer_type(gl.int32), bitcast=True)
    TW: gl.constexpr = 1792                                   # words per 7,168-byte tile
    tile0 = (pid_n * num_k_blocks).to(gl.int64) * TW
    roff = cr.to(gl.int64) * 28
    # the activation tile: 16 rows x 64 words per k-block, 4-byte cp.async, rows >= M masked
    cpa: gl.constexpr = BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0])
    am = gl.arange(0, BM, layout=SliceLayout(1, cpa))
    aw = gl.arange(0, 64, layout=SliceLayout(0, cpa))
    XQ32 = XQ.to(gl.pointer_type(gl.int32), bitcast=True)
    arow = am.to(gl.int64) * (stride_xq // 4)
    amask = (am < M)[:, None] & (aw < 64)[None, :]

    bidx = gl.arange(0, 4, layout=SliceLayout(0, L4))
    zero4 = gl.zeros([64, 4], gl.int32, L4)
    b4 = zero4 + bidx[None, :]
    selq = b4 | 0x4440
    shq_lo = 8 - b4
    shq_hi = 4 - b4
    shs_lo = 4 * b4
    shs_hi = 16 + 4 * b4
    tbase = _smem_base(zero4) + TAB_OFF
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    src0 = WT32 + tile0 + kb0 * TW + roff[:, None] + cc[None, :]
    async_copy.async_copy_global_to_shared(smem.index(0), src0, cmask)
    async_copy.async_copy_global_to_shared(smem_a.index(0), XQ32 + arow[:, None] + (kb0 * 64 + aw)[None, :], amask)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            src1 = WT32 + tile0 + (kb + 1) * TW + roff[:, None] + cc[None, :]
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), src1, cmask)
            async_copy.async_copy_global_to_shared(smem_a.index((it + 1) % 2), XQ32 + arow[:, None] + ((kb + 1) * 64 + aw)[None, :], amask)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        words = smem.index(it % 2)
        a_tile = smem_a.index(it % 2)._reinterpret(gl.int8, [BM, 256], smem_a8)
        qh01 = _rowword_in(words, 16, Lcol_w, Lrow4)
        qh23 = _rowword_in(words, 17, Lcol_w, Lrow4)
        dw = _rowword_in(words, 27, Lcol_c, Lrow_c)
        d = (dw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        scw = _rowword_in(words, 26, Lcol_c, Lrow_c)
        sx = gl.load(sxrow + kb, mask=ms_ok, other=0.0)                           # the activation scale of this k-block (per 256)
        acc_k = gl.zeros([BM, BN], gl.int32, mma)
        for ib in gl.static_range(8):
            a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)
            qsw0 = _rowword_in(words, 2 * ib, Lcol_w, Lrow4)
            qsw1 = _rowword_in(words, 2 * ib + 1, Lcol_w, Lrow4)
            sgw = _rowword_in(words, 18 + ib, Lcol_w, Lrow4)
            if ib < 4:
                qh = (qh01 >> (8 * ib)) & 0xFF
            else:
                qh = (qh23 >> (8 * (ib - 4))) & 0xFF
            wq_lo = _decode_half_t(qsw0, qh, sgw, selq, shq_lo, shs_lo, zero4, tbase)
            wq_hi = _decode_half_t(qsw1, qh, sgw, selq, shq_hi, shs_hi, zero4, tbase)
            w8 = gl.join(wq_lo, wq_hi)
            w4 = _word_to_i8x4(gl.join(gl.join(w8, w8), gl.join(w8, w8)))
            b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (2, 1, 3, 4, 0)), [32, 64]), db8)
            acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
            si = 1 + 2 * ((scw >> (4 * ib)) & 0xF)                                # the sub-block scale as the integer it is
            acc_k = acc_k + acc_i * si[None, :]
        acc = acc + acc_k.to(gl.float32) * (sx[:, None] * d[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    keep = gtab.slice(0, 64).load(SliceLayout(0, mma))
    omask = omask & (keep[None, :] != 0x7FFFFFFF)
    if REDUCE and SPLITK > 1:
        # the fp32 partial to P[pid_k]; the last CTA of the tile sums the partials in order into Y (bf16)
        _splitk_reduce(Y, P, C, acc, ym, yn, omask, pid_n, pid_k, stride_ym, stride_pm, stride_pk, SPLITK)
    else:
        Yp = Y + pid_k.to(gl.int64) * stride_yk
        gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def iq3s_int8te_gemm(WT: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                    quantized: tuple | None = None, tab_off: int = TAB_OFF, reduce: bool = False, return_partials: bool = False, out: torch.Tensor | None = None):
    """WT the tile-major repack [n_tiles, nb, 64, 112] of the rows."""
    BN = 64
    assert WT.dtype == torch.uint8 and WT.is_cuda and WT.dim() == 4 and WT.is_contiguous() and WT.shape[2] == 64 and WT.shape[3] == TILE_ROW_BYTES
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    nb = K // 256
    assert K % 256 == 0 and WT.shape[1] == nb and WT.shape[0] == -(-n_out // BN) and M <= 16
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations_256(X.contiguous())   # (int8 X, scales per 256)
    assert XQ.stride(0) % 4 == 0 and XQ.data_ptr() % 4 == 0
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=TILE_ROW_BYTES)
    splitk = max(1, min(splitk, nb))
    Y, P, C, strides = splitk_buffers(X, n_out, splitk, reduce, out)
    iq3s_int8te_kernel[(WT.shape[0], splitk)](
        XQ, SX, WT, Y, grid32(WT.device), M, n_out, nb, XQ.stride(0), SX.stride(0), strides[0], strides[1], P, C, strides[2], strides[3],
        BM=16, BN=BN, SPLITK=splitk, REDUCE=bool(reduce and splitk > 1), TAB_OFF=tab_off, num_warps=num_warps)
    res = splitk_result(Y, X, splitk, reduce, out)
    return (res, P) if return_partials else res
