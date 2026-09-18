"""IQ2_XXS / IQ2_XS / IQ2_S x bf16 GEMM in Gluon, the int8 form: the IQ3_S int8 kernel with the IQ2 block decoding of the bf16
kernel (constexpr type switch). A B-fragment word b of a
sub-block (weights 4 b .. 4 b + 3) is word b mod 2 of the 8-byte lattice
entry of group b // 2 - the four magnitudes 8 / 25 / 43 / 67 as int8 straight
from the gather - and bits 4 (b mod 2) .. + 3 of the group's 8-bit sign
mask, spread to byte masks and applied by the byte-wise negation.

The scale: IQ2_XXS has one per sub-block of 32 (bits 28..31 of the aux
word), so one mma.m16n8k32.s8 per sub-block as in the IQ3_S form. IQ2_XS
and IQ2_S carry one per 16 weights (a nibble each, groups 0 / 1 the low
nibble, 2 / 3 the high), and a k32 mma sums both halves into one int32:
so two mmas per sub-block, each on the B operand with the other half's
words zeroed, two int32 accumulators, each scaled by its own d (2 s + 1) / 8
in the fp32 epilogue. The mma is not the bound of these kernels (one byte
of weights per instruction is), so the second mma costs little; the
alternative, m16n8k16.s8 on half-width fragments, needs a second operand
layout that this kernel family has not measured.

The weight = d (2 s + 1) / 8 x magnitude x sign; the activations quantised
per 32 by quantize_activations (the plugin's MMVQ scheme). Staging and the
straddle fields as in the bf16 kernel (2-byte-aligned blocks of 66 / 74 /
82 bytes).
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

from .iq2 import BLOCK_BYTES, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, _popc, grid_words, split_k_for
from .iq3s_int8 import _field_in, _rowword_in, _word_to_i8x4, quantize_activations


@g.jit
def _negate_bytes(gw, nib):
    """Four int8 magnitudes in gw negated where the four bits of nib say."""
    mask4 = ((nib * 0x204081) & 0x01010101) * 0xFF
    return (gw ^ mask4) + (mask4 & 0x01010101)


@g.jit
def _b_operand(wq, db8: gl.constexpr):
    w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
    return gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)


@g.jit
def iq2_int8_kernel(XQ, SX, W, Y, GRID,
                    M, N, num_k_blocks, row_bytes, stride_xq, stride_sx, stride_ym, stride_yk, w_shift, w_end,
                    TYPE: gl.constexpr, BLOCK: gl.constexpr, WORDS: gl.constexpr,
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
    o0 = crow + kb0 * BLOCK
    src0 = (o0 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
    m0 = (cn < N)[:, None] & (cw < WORDS)[None, :] & (src0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem.index(0), (W + src0).to(gl.pointer_type(gl.int32), bitcast=True), m0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            o1 = crow + (kb + 1) * BLOCK
            src1 = (o1 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
            m1 = (cn < N)[:, None] & (cw < WORDS)[None, :] & (src1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), (W + src1).to(gl.pointer_type(gl.int32), bitcast=True), m1)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        words = smem.index(it % 2)
        s2w = ((drow_w + kb * BLOCK) & 2).to(gl.int32)
        selw = gl.where(s2w == 0, 0x5432, 0x7654)
        s2c = ((drow_c + kb * BLOCK) & 2).to(gl.int32)
        selc = gl.where(s2c == 0, 0x5432, 0x7654)
        dw = _rowword_in(words, 0, Lcol_c, Lrow_c)
        d = ((dw >> (8 * s2c)) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            if TYPE == 16:      # IQ2_XXS: one scale per sub-block, one mma
                f0 = _field_in(words, 2 * ib, selw, Lcol_w, Lrow_w)             # the four 8-bit grid indices (block byte 2 + 8 ib)
                aux = _field_in(words, 2 * ib + 1, selw, Lcol_w, Lrow_w)        # signs (7 bits per group) | scale << 28
                auxc = _field_in(words, 2 * ib + 1, selc, Lcol_c, Lrow_c)       # the same word for the scale, column layout
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
                if TYPE == 17:  # IQ2_XS: u16 per group (9-bit index, 7-bit signs); scales[ib] at block byte 66 + ib
                    f0 = _field_in(words, 2 * ib, selw, Lcol_w, Lrow_w)         # u16 of groups 0, 1
                    f1 = _field_in(words, 2 * ib + 1, selw, Lcol_w, Lrow_w)     # u16 of groups 2, 3
                    scb = (_field_in(words, 16 + ib // 4, selc, Lcol_c, Lrow_c) >> (8 * (ib % 4))) & 0xFF
                    f0_2 = zero2 + f0[:, None]
                    f1_2 = zero2 + f1[:, None]
                    q2 = gl.where(g2 < 2, f0_2 >> (16 * (g2 & 1)), f1_2 >> (16 * (g2 & 1))) & 0xFFFF
                    idx = q2 & 511
                    s7 = q2 >> 9
                    mask8 = s7 | ((_popc(s7) & 1) << 7)
                else:           # IQ2_S (22): index bytes, sign bytes, qh bits 8..9, scales[ib] at 74 + ib
                    f0 = _field_in(words, ib, selw, Lcol_w, Lrow_w)             # the four 8-bit grid indices (block byte 2 + 4 ib)
                    sg = _field_in(words, 8 + ib, selw, Lcol_w, Lrow_w)         # the four sign bytes (34 + 4 ib)
                    qhb = (_field_in(words, 16 + ib // 4, selw, Lcol_w, Lrow_w) >> (8 * (ib % 4))) & 0xFF   # qh[ib] (66 + ib)
                    scb = (_field_in(words, 18 + ib // 4, selc, Lcol_c, Lrow_c) >> (8 * (ib % 4))) & 0xFF   # scales[ib] (74 + ib)
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
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def iq2_int8_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, quant_type: int, splitk: int | None = None,
                  num_warps: int = 4, quantized: tuple | None = None) -> torch.Tensor:
    """W: the raw rows of the type (uint8 [n_out, K/256*bytes], any row stride); X bf16 [M <= 16, K];
    quantized: (int8 X, scales[, sums]) from quantize_activations, else quantised here."""
    BN = 64
    block = BLOCK_BYTES[quant_type]
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * block and M <= 16
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations(X.contiguous())
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=block)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    off = W.storage_offset()
    base = torch.as_strided(W, (1,), (1,), off & -4)
    w_shift = off & 3
    w_end = W.untyped_storage().nbytes() - (off & -4)
    iq2_int8_kernel[(triton.cdiv(n_out, BN), splitk)](
        XQ, SX, base, Y, grid_words(quant_type, W.device), M, n_out, nb, W.stride(0), XQ.stride(0), SX.stride(0),
        Y.stride(1), Y.stride(0), w_shift, w_end,
        TYPE=quant_type, BLOCK=block, WORDS=(block + 2 + 3) // 4, BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]


def iq2xxs_int8_gemm(W, X, n_out, splitk=None, num_warps=4, quantized=None):
    return iq2_int8_gemm(W, X, n_out, GGML_TYPE_IQ2_XXS, splitk, num_warps, quantized)


def iq2xs_int8_gemm(W, X, n_out, splitk=None, num_warps=4, quantized=None):
    return iq2_int8_gemm(W, X, n_out, GGML_TYPE_IQ2_XS, splitk, num_warps, quantized)


def iq2s_int8_gemm(W, X, n_out, splitk=None, num_warps=4, quantized=None):
    return iq2_int8_gemm(W, X, n_out, GGML_TYPE_IQ2_S, splitk, num_warps, quantized)
