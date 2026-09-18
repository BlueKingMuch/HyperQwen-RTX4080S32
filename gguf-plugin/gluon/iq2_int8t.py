"""IQ2_XXS / IQ2_XS / IQ2_S x bf16 GEMM in Gluon, the int8 form on a
tile-major repack with 16-byte copies, carrying the constexpr type switch
of the non-tiled int8 form. The
block's bytes after the 2-byte d are laid out per row as words 0.., d in the
low half of the word after them (IQ2_XXS 66 -> 68 bytes, 17 words; IQ2_XS
74 -> 76, 19; IQ2_S 82 -> 84, 21 - all odd, conflict-free column reads), so
the row form's field k (block byte 2 + 4 k) is tile word k and the decode
is the int8 form's unchanged: bit-identical. A tile's words (1,088 / 1,216 /
1,344) are copied linearly in 16-byte chunks (one masked chunk of 2,048)
into a flat [2, 2048]-word shared array; the fields are ld.shared gathers.

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

from .iq2 import BLOCK_BYTES, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, _popc, grid_words, split_k_for
from .iq2_int8 import _b_operand, _negate_bytes
from .iq3s_int8 import quantize_activations
from .iq3s_int8t import _smem_base
from .iq3xxs_int8t import repack_dlast_tiles, tile_geometry
from .iq4xs_int8t import _lds_v


@g.jit
def iq2_int8t_kernel(XQ, SX, WT, Y, GRID,
                     M, N, num_k_blocks, stride_xq, stride_sx, stride_ym, stride_yk, P, C, stride_pm, stride_pk,
                     TYPE: gl.constexpr, DW: gl.constexpr,
                     BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr, REDUCE: gl.constexpr, ROW_WORDS: gl.constexpr,
                     TW: gl.constexpr, NCH: gl.constexpr, SLOT: gl.constexpr, SMEM_OFF: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_flat: gl.constexpr = SwizzledSharedLayout(4, 1, 1, [0])
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 4], [32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 8])
    Lrow_w: gl.constexpr = SliceLayout(1, Lw)
    Lrow_c: gl.constexpr = SliceLayout(0, mma)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    c16: gl.constexpr = BlockedLayout([4], [32], [4], [0])
    im = gl.arange(0, 2048, layout=c16)
    im = gl.max_contiguous(gl.multiple_of(im, 4), 4)
    WT32 = WT.to(gl.pointer_type(gl.int32), bitcast=True)
    tile0 = (pid_n * num_k_blocks).to(gl.int64) * TW
    smem = gl.allocate_shared_memory(gl.int32, [2, SLOT], smem_flat)

    rw = gl.arange(0, 64, layout=Lrow_w)
    aw = _smem_base(rw) + SMEM_OFF + rw * (ROW_WORDS * 4)
    rc = gl.arange(0, 64, layout=Lrow_c)
    ac = _smem_base(rc) + SMEM_OFF + rc * (ROW_WORDS * 4)
    bidx = gl.arange(0, 8, layout=SliceLayout(0, Lw))
    zero2 = gl.zeros([64, 8], gl.int32, Lw)
    b2 = zero2 + bidx[None, :]
    g2 = b2 >> 1                                # the word's group (8 weights, one lattice entry)
    w2 = b2 & 1                                 # the word within the entry's 8 bytes
    nsh2 = 4 * w2                               # the word's nibble of the group's 8-bit sign mask
    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    base0 = WT32 + tile0 + kb0 * TW
    for c in gl.static_range(NCH):
        if (c + 1) * 2048 <= TW:
            async_copy.async_copy_global_to_shared(smem.index(0).slice(c * 2048, 2048), base0 + c * 2048 + im)
        else:
            async_copy.async_copy_global_to_shared(smem.index(0).slice(c * 2048, 2048), base0 + c * 2048 + im, im < (TW - c * 2048))
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            base1 = WT32 + tile0 + (kb + 1) * TW
            for c in gl.static_range(NCH):
                if (c + 1) * 2048 <= TW:
                    async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(c * 2048, 2048), base1 + c * 2048 + im)
                else:
                    async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(c * 2048, 2048), base1 + c * 2048 + im, im < (TW - c * 2048))
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        so = (it % 2) * (SLOT * 4)
        dw = _lds_v(ac + so + DW * 4)
        d = (dw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            if TYPE == 16:      # IQ2_XXS: one scale per sub-block, one mma
                f0 = _lds_v(aw + so + (2 * ib) * 4)                              # the four 8-bit grid indices
                aux = _lds_v(aw + so + (2 * ib + 1) * 4)                         # signs (7 bits per group) | scale << 28
                auxc = _lds_v(ac + so + (2 * ib + 1) * 4)                        # the same word for the scale, column layout
                f0_2 = zero2 + f0[:, None]
                aux_2 = zero2 + aux[:, None]
                idx = (f0_2 >> (8 * g2)) & 0xFF
                s7 = (aux_2 >> (7 * g2)) & 127
                mask8 = s7 | ((_popc(s7) & 1) << 7)
                gw = gl.load(GRID + idx * 2 + w2)
                wq = _negate_bytes(gw, (mask8 >> nsh2) & 0xF)
                acc_i = mma_v2(a8, _b_operand(wq, db8), gl.zeros([BM, BN], gl.int32, mma))
                s = (auxc >> 28) & 0xF
                dl = d * (2.0 * s.to(gl.float32) + 1.0) * 0.125
                acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
            else:
                if TYPE == 17:  # IQ2_XS: u16 per group (9-bit index, 7-bit signs); scales[ib] at word 16 + ib // 4
                    f0 = _lds_v(aw + so + (2 * ib) * 4)                          # u16 of groups 0, 1
                    f1 = _lds_v(aw + so + (2 * ib + 1) * 4)                      # u16 of groups 2, 3
                    scb = (_lds_v(ac + so + (16 + ib // 4) * 4) >> (8 * (ib % 4))) & 0xFF
                    f0_2 = zero2 + f0[:, None]
                    f1_2 = zero2 + f1[:, None]
                    q2 = gl.where(g2 < 2, f0_2 >> (16 * (g2 & 1)), f1_2 >> (16 * (g2 & 1))) & 0xFFFF
                    idx = q2 & 511
                    s7 = q2 >> 9
                    mask8 = s7 | ((_popc(s7) & 1) << 7)
                else:           # IQ2_S (22): index bytes (words 0-7), sign bytes (8-15), qh (16-17), scales (18-19)
                    f0 = _lds_v(aw + so + ib * 4)                                # the four 8-bit grid indices
                    sg = _lds_v(aw + so + (8 + ib) * 4)                          # the four sign bytes
                    qhb = (_lds_v(aw + so + (16 + ib // 4) * 4) >> (8 * (ib % 4))) & 0xFF        # qh[ib]
                    scb = (_lds_v(ac + so + (18 + ib // 4) * 4) >> (8 * (ib % 4))) & 0xFF        # scales[ib]
                    f0_2 = zero2 + f0[:, None]
                    sg_2 = zero2 + sg[:, None]
                    qh_2 = zero2 + qhb[:, None]
                    idx = ((f0_2 >> (8 * g2)) & 0xFF) | (((qh_2 >> (2 * g2)) & 3) << 8)
                    mask8 = (sg_2 >> (8 * g2)) & 0xFF
                gw = gl.load(GRID + idx * 2 + w2)
                wq = _negate_bytes(gw, (mask8 >> nsh2) & 0xF)
                # one scale per 16 weights: the two halves on their own mma, the other half's words zeroed
                wq_lo = gl.where(b2 < 4, wq, 0)
                wq_hi = gl.where(b2 >= 4, wq, 0)
                acc_lo = mma_v2(a8, _b_operand(wq_lo, db8), gl.zeros([BM, BN], gl.int32, mma))
                acc_hi = mma_v2(a8, _b_operand(wq_hi, db8), gl.zeros([BM, BN], gl.int32, mma))
                s_lo = scb & 0xF
                s_hi = (scb >> 4) & 0xF
                dl_lo = d * (2.0 * s_lo.to(gl.float32) + 1.0) * 0.125
                dl_hi = d * (2.0 * s_hi.to(gl.float32) + 1.0) * 0.125
                acc = acc + acc_lo.to(gl.float32) * (sx[:, None] * dl_lo[None, :]) + acc_hi.to(gl.float32) * (sx[:, None] * dl_hi[None, :])
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


def repack_iq2_tiles(W: torch.Tensor, n_out: int, quant_type: int, row_bytes: int | None = None) -> torch.Tensor:
    """Raw rows of the type -> tile-major [n_tiles, nb, 64, row_bytes] with d behind the block's other bytes
    (row_bytes default: the block rounded up to a word after the move, 68 / 76 / 84)."""
    block = BLOCK_BYTES[quant_type]
    if row_bytes is None:
        row_bytes = (block + 2 + 3) // 4 * 4
    return repack_dlast_tiles(W, n_out, block, row_bytes)


def iq2_int8t_gemm(WT: torch.Tensor, X: torch.Tensor, n_out: int, quant_type: int, splitk: int | None = None,
                   num_warps: int = 4, quantized: tuple | None = None, smem_off: int = 0, reduce: bool = False, return_partials: bool = False, out: torch.Tensor | None = None):
    """WT the tile-major repack (repack_iq2_tiles); quantized: (int8 X, scales[, sums])."""
    BN = 64
    block = BLOCK_BYTES[quant_type]
    assert WT.dtype == torch.uint8 and WT.is_cuda and WT.dim() == 4 and WT.is_contiguous() and WT.shape[2] == 64
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    nb = K // 256
    row_bytes = WT.shape[3]
    assert K % 256 == 0 and WT.shape[1] == nb and WT.shape[0] == -(-n_out // BN) and M <= 16 and row_bytes >= block + 2
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations(X.contiguous())
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=row_bytes)   # 0034: the split capped by the batch rows
    splitk = max(1, min(splitk, nb))
    Y, P, C, strides = splitk_buffers(X, n_out, splitk, reduce, out)
    row_words, tw, nch, slot = tile_geometry(row_bytes)
    iq2_int8t_kernel[(WT.shape[0], splitk)](
        XQ, SX, WT, Y, grid_words(quant_type, WT.device), M, n_out, nb, XQ.stride(0), SX.stride(0), strides[0], strides[1], P, C, strides[2], strides[3],
        TYPE=quant_type, DW=(block - 2) // 4, BM=16, BN=BN, SPLITK=splitk, REDUCE=bool(reduce and splitk > 1), ROW_WORDS=row_words, TW=tw, NCH=nch, SLOT=slot,
        SMEM_OFF=smem_off, num_warps=num_warps)
    res = splitk_result(Y, X, splitk, reduce, out)
    return (res, P) if return_partials else res
