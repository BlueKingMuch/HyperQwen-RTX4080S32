"""Q4_K x bf16 GEMM in Gluon, the int8 form on a tile-major repack with
16-byte copies, in the pattern of the IQ4_XS tiled form. The rows
of 144 bytes (36 words, word-aligned already: d | dmin in word 0, the scale
and min bytes in words 1-3, qs in words 4-35) are laid out tile-major and
copied linearly in 16-byte chunks into a flat [2, 4096]-word shared array;
the header words are read as ld.shared gathers per row, a group's eight qs
words as a [64, 8] gather in the decode layout. The decode (nibbles as
unsigned int8), the min term through the activation group's sum, the direct
A loads, the mma and the epilogue are the int8 form's: bit-identical.

The split-K reduction inside the kernel is an option (reduce=; +2.5 % on C1
in situ, so the sum kernel stays the default), and the output view is out=.
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

from .splitk_reduce import _splitk_reduce, splitk_buffers, splitk_result   # 0039: the in-kernel split-K reduction

from .iq3s import split_k_for
from .iq3s_int8 import _word_to_i8x4, quantize_activations
from .iq3s_int8t import _smem_base
from .iq4xs_int8t import _lds_v
from .q4k_int8 import BLOCK_BYTES


@g.jit
def q4k_int8t_kernel(XQ, SX, SUMX, WT, Y,
                     M, N, num_k_blocks, stride_xq, stride_sx, stride_ym, stride_yk, P, C, stride_pm, stride_pk,
                     BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr, REDUCE: gl.constexpr, ROW_WORDS: gl.constexpr,
                     TW: gl.constexpr, NMAIN: gl.constexpr, RP: gl.constexpr, SLOT: gl.constexpr, SMEM_OFF: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_flat: gl.constexpr = SwizzledSharedLayout(4, 1, 1, [0])
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 4], [32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 8])
    Lrow_c: gl.constexpr = SliceLayout(0, mma)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    c16: gl.constexpr = BlockedLayout([4], [32], [4], [0])
    c1: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    im = gl.arange(0, 2048, layout=c16)
    im = gl.max_contiguous(gl.multiple_of(im, 4), 4)
    ir = gl.arange(0, RP, layout=c1)
    rmask = ir < (TW - NMAIN)
    WT32 = WT.to(gl.pointer_type(gl.int32), bitcast=True)
    tile0 = (pid_n * num_k_blocks).to(gl.int64) * TW
    smem = gl.allocate_shared_memory(gl.int32, [2, SLOT], smem_flat)

    # the field addresses: the header per row in the accumulator's column layout, a group's 8 words as [64, 8] in the decode layout
    rc = gl.arange(0, 64, layout=Lrow_c)
    ac = _smem_base(rc) + SMEM_OFF + rc * (ROW_WORDS * 4)
    r2 = gl.arange(0, 64, layout=SliceLayout(1, Lw))
    c2 = gl.arange(0, 8, layout=SliceLayout(0, Lw))
    a2 = _smem_base(r2)[:, None] + SMEM_OFF + (r2[:, None] * ROW_WORDS + 4 + c2[None, :]) * 4

    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx
    sumrow = SUMX + ms * stride_sx

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    base0 = WT32 + tile0 + kb0 * TW
    for c in gl.static_range(NMAIN // 2048):
        async_copy.async_copy_global_to_shared(smem.index(0).slice(c * 2048, 2048), base0 + c * 2048 + im)
    if RP > 0:
        async_copy.async_copy_global_to_shared(smem.index(0).slice(NMAIN, RP), base0 + NMAIN + ir, rmask)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            base1 = WT32 + tile0 + (kb + 1) * TW
            for c in gl.static_range(NMAIN // 2048):
                async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(c * 2048, 2048), base1 + c * 2048 + im)
            if RP > 0:
                async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(NMAIN, RP), base1 + NMAIN + ir, rmask)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        so = (it % 2) * (SLOT * 4)
        w0 = _lds_v(ac + so)                                                     # d | dmin << 16
        sw1 = _lds_v(ac + so + 4)
        sw2 = _lds_v(ac + so + 8)
        sw3 = _lds_v(ac + so + 12)
        d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        dmin = ((w0 >> 16) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for gidx in gl.static_range(4):
            qw = _lds_v(a2 + so + gidx * 32)                                    # the group's 8 words, [64, 8] in the decode layout
            for sub in gl.static_range(2):
                ib = 2 * gidx + sub
                a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
                sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                sumx = gl.load(sumrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                if ib < 4:
                    sc = (sw1 >> (8 * ib)) & 63
                    mn = (sw2 >> (8 * ib)) & 63
                else:
                    sc = ((sw3 >> (8 * (ib - 4))) & 0xF) | (((sw1 >> (8 * (ib - 4) + 6)) & 3) << 4)
                    mn = ((sw3 >> (8 * (ib - 4) + 4)) & 0xF) | (((sw2 >> (8 * (ib - 4) + 6)) & 3) << 4)
                wq = (qw >> (4 * sub)) & 0x0F0F0F0F                               # four unsigned 4-bit weights
                w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
                b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
                acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
                dl = d * sc.to(gl.float32)
                ml = dmin * mn.to(gl.float32)
                acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :]) - sumx[:, None] * ml[None, :]
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    keep = smem.index(0).slice(0, 64).load(SliceLayout(0, mma))
    omask = omask & (keep[None, :] != 0x7FFFFFFF)
    if REDUCE and SPLITK > 1:
        # the fp32 partial to P[pid_k]; the last CTA of the tile sums the partials in order into Y (bf16)
        _splitk_reduce(Y, P, C, acc, ym, yn, omask, pid_n, pid_k, stride_ym, stride_pm, stride_pk, SPLITK)
    else:
        Yp = Y + pid_k.to(gl.int64) * stride_yk
        gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def repack_q4k_tiles(W: torch.Tensor, n_out: int, row_bytes: int = 144) -> torch.Tensor:
    """Raw Q4_K rows [n_out, nb * 144] (uint8) -> tile-major uint8 [n_tiles, nb, 64, row_bytes], rows past n_out zero."""
    assert W.dtype == torch.uint8 and W.dim() == 2 and W.shape[0] == n_out and W.shape[1] % BLOCK_BYTES == 0 and row_bytes >= BLOCK_BYTES and row_bytes % 4 == 0
    nb = W.shape[1] // BLOCK_BYTES
    n_tiles = -(-n_out // 64)
    t = W.reshape(n_out, nb, BLOCK_BYTES)
    if row_bytes > BLOCK_BYTES:
        t = torch.cat([t, torch.zeros((n_out, nb, row_bytes - BLOCK_BYTES), dtype=torch.uint8, device=W.device)], dim=-1)
    if n_tiles * 64 > n_out:
        t = torch.cat([t, torch.zeros((n_tiles * 64 - n_out, nb, row_bytes), dtype=torch.uint8, device=W.device)], dim=0)
    return t.reshape(n_tiles, 64, nb, row_bytes).permute(0, 2, 1, 3).contiguous()


def q4k_int8t_gemm(WT: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                   quantized: tuple | None = None, smem_off: int = 0, reduce: bool = False, return_partials: bool = False, out: torch.Tensor | None = None):
    """WT the tile-major repack (repack_q4k_tiles); quantized: (int8 X, scales, group sums)."""
    BN = 64
    assert WT.dtype == torch.uint8 and WT.is_cuda and WT.dim() == 4 and WT.is_contiguous() and WT.shape[2] == 64
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    nb = K // 256
    row_bytes = WT.shape[3]
    assert K % 256 == 0 and WT.shape[1] == nb and WT.shape[0] == -(-n_out // BN) and M <= 16 and row_bytes % 4 == 0
    if quantized is not None and len(quantized) == 3:
        XQ, SX, SUMX = quantized
    else:
        XQ, SX, SUMX = quantize_activations(X.contiguous(), with_sums=True)
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=row_bytes)   # 0034: the split capped by the batch rows
    splitk = max(1, min(splitk, nb))
    Y, P, C, strides = splitk_buffers(X, n_out, splitk, reduce, out)
    tw = 64 * row_bytes // 4
    slot = 1 << (tw - 1).bit_length()
    nmain = (tw // 2048) * 2048
    rem = tw - nmain
    rp = 0 if rem == 0 else 1 << (rem - 1).bit_length()
    q4k_int8t_kernel[(WT.shape[0], splitk)](
        XQ, SX, SUMX, WT, Y, M, n_out, nb, XQ.stride(0), SX.stride(0), strides[0], strides[1], P, C, strides[2], strides[3],
        BM=16, BN=BN, SPLITK=splitk, REDUCE=bool(reduce and splitk > 1), ROW_WORDS=row_bytes // 4, TW=tw, NMAIN=nmain, RP=rp, SLOT=slot, SMEM_OFF=smem_off, num_warps=num_warps)
    res = splitk_result(Y, X, splitk, reduce, out)
    return (res, P) if return_partials else res
