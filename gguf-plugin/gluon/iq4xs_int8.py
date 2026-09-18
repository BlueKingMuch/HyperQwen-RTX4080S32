"""IQ4_XS x bf16 GEMM in Gluon, the int8 form: the
nibbles of a word mapped through kvalues_iq4nl into four signed int8
weights in one register - the B fragment of mma.m16n8k32.s8 - with no
table gather at all: the 16-entry value table sits in four constant
registers and two prmt plus a byte select do the lookup.

Per B-fragment word b of a sub-block (k = 4 b .. 4 b + 3): qs word b mod 4
of the sub-block, low nibbles for b < 4 and high for b >= 4; the four
nibbles t (one per byte) select from kvalues by prmt(T0, T1, t & 7) for
entries 0..7 and prmt(T2, T3, t & 7) for 8..15, chosen per byte by bit 3
(the four 3-bit selectors compressed into prmt's nibble form first).
The sub-block scale d (ls - 32) is applied in the fp32 epilogue with the
token's activation scale (quantize_activations of the IQ3_S int8 kernel).
The word-aligned 136-byte blocks are staged as in the bf16 kernel (two
word arrays: the 32 qs words and the 2 header words per row).
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

from .iq3s import _prmt, split_k_for
from .iq3s_int8 import _rowword_in, _word_to_i8x4, quantize_activations

BLOCK_BYTES = 136
KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _pack4(vals):
    return int.from_bytes(bytes((v & 0xFF) for v in vals), "little", signed=True)


# kvalues as four packed byte registers (entries 0..3, 4..7, 8..11, 12..15)
T0, T1, T2, T3 = (_pack4(KVALUES[i:i + 4]) for i in (0, 4, 8, 12))


@g.jit
def _kvalues_lookup(nib4, T0: gl.constexpr, T1: gl.constexpr, T2: gl.constexpr, T3: gl.constexpr):
    """Four nibble indices (one per byte of nib4) -> four int8 kvalues packed."""
    # prmt reads the selector of output byte i from nibble i (bits 4 i .. 4 i + 3):
    # compress the four 3-bit indices, one per byte of nib4, into the low 16 bits
    sel8 = nib4 & 0x07070707
    sel4 = (sel8 | (sel8 >> 4)) & 0x00FF00FF
    sel = (sel4 | (sel4 >> 8)) & 0xFFFF
    lo = _prmt(gl.full_like(nib4, T0), gl.full_like(nib4, T1), sel)
    hi = _prmt(gl.full_like(nib4, T2), gl.full_like(nib4, T3), sel)
    m = ((nib4 >> 3) & 0x01010101) * 0xFF
    return (lo & ~m) | (hi & m)


@g.jit
def iq4xs_int8_kernel(XQ, SX, W, Y,
                      M, N, num_k_blocks, row_bytes, stride_xq, stride_sx, stride_ym, stride_yk, w_end,
                      T0: gl.constexpr, T1: gl.constexpr, T2: gl.constexpr, T3: gl.constexpr,
                      BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_qs: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    smem_hd: gl.constexpr = SwizzledSharedLayout(1, 1, 1, [1, 0])
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

    cq: gl.constexpr = BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    ch: gl.constexpr = BlockedLayout([1, 1], [16, 2], [4, 1], [1, 0])
    cn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, cq))
    cw = gl.arange(0, 32, layout=SliceLayout(0, cq))
    hn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, ch))
    hw = gl.arange(0, 2, layout=SliceLayout(0, ch))
    crow = cn.to(gl.int64) * row_bytes
    hrow = hn.to(gl.int64) * row_bytes
    smem_q = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_qs)
    smem_h = gl.allocate_shared_memory(gl.int32, [2, 64, 2], smem_hd)

    bidx = gl.arange(0, 8, layout=SliceLayout(0, Lw))
    zero2 = gl.zeros([64, 8], gl.int32, Lw)
    b2 = zero2 + bidx[None, :]
    nsh2 = (b2 >> 2) * 4                                   # 0 for the low nibbles, 4 for the high
    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    src_q0 = (crow + kb0 * 136 + 8)[:, None] + 4 * cw[None, :].to(gl.int64)
    src_h0 = (hrow + kb0 * 136)[:, None] + 4 * hw[None, :].to(gl.int64)
    mq0 = (cn < N)[:, None] & (src_q0 + 4 <= w_end)
    mh0 = (hn < N)[:, None] & (src_h0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem_q.index(0), (W + src_q0).to(gl.pointer_type(gl.int32), bitcast=True), mq0)
    async_copy.async_copy_global_to_shared(smem_h.index(0), (W + src_h0).to(gl.pointer_type(gl.int32), bitcast=True), mh0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            src_q1 = (crow + (kb + 1) * 136 + 8)[:, None] + 4 * cw[None, :].to(gl.int64)
            src_h1 = (hrow + (kb + 1) * 136)[:, None] + 4 * hw[None, :].to(gl.int64)
            mq1 = (cn < N)[:, None] & (src_q1 + 4 <= w_end)
            mh1 = (hn < N)[:, None] & (src_h1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem_q.index((it + 1) % 2), (W + src_q1).to(gl.pointer_type(gl.int32), bitcast=True), mq1)
            async_copy.async_copy_global_to_shared(smem_h.index((it + 1) % 2), (W + src_h1).to(gl.pointer_type(gl.int32), bitcast=True), mh1)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        qwords = smem_q.index(it % 2)
        hwords = smem_h.index(it % 2)
        # per row in the accumulator's column layout: d, scales_h, scales_l
        w0 = _rowword_in(hwords, 0, Lcol_c, Lrow_c)
        slw = _rowword_in(hwords, 1, Lcol_c, Lrow_c)
        d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        shw = (w0 >> 16) & 0xFFFF
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            # the sub-block's four qs words per row, the word b mod 4 per element
            q0 = _rowword_in(qwords, 4 * ib, Lcol_w, Lrow_w)
            q1 = _rowword_in(qwords, 4 * ib + 1, Lcol_w, Lrow_w)
            q2 = _rowword_in(qwords, 4 * ib + 2, Lcol_w, Lrow_w)
            q3 = _rowword_in(qwords, 4 * ib + 3, Lcol_w, Lrow_w)
            q0_2 = zero2 + q0[:, None]
            q1_2 = zero2 + q1[:, None]
            q2_2 = zero2 + q2[:, None]
            q3_2 = zero2 + q3[:, None]
            bl = b2 & 3
            qw = gl.where(bl == 0, q0_2, gl.where(bl == 1, q1_2, gl.where(bl == 2, q2_2, q3_2)))
            nib4 = (qw >> nsh2) & 0x0F0F0F0F                                  # four nibble indices, one per byte
            wq = _kvalues_lookup(nib4, T0, T1, T2, T3)                          # four signed int8 weights
            w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
            b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
            acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
            ls = ((slw >> (4 * ib)) & 0xF) | (((shw >> (2 * ib)) & 3) << 4)
            dl = d * (ls.to(gl.float32) - 32.0)
            acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def iq4xs_int8_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                    quantized: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    BN = 64
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * BLOCK_BYTES and M <= 16
    if W.stride(0) % 4 != 0 or W.data_ptr() % 4 != 0:
        W = W.contiguous()
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations(X.contiguous())
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=BLOCK_BYTES)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    w_end = W.untyped_storage().nbytes() - W.storage_offset()
    iq4xs_int8_kernel[(triton.cdiv(n_out, BN), splitk)](
        XQ, SX, W, Y, M, n_out, nb, W.stride(0), XQ.stride(0), SX.stride(0), Y.stride(1), Y.stride(0), w_end,
        T0=T0, T1=T1, T2=T2, T3=T3, BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
