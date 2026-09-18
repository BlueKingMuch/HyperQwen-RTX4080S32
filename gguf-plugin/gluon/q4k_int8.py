"""Q4_K x bf16 GEMM in Gluon, the int8 form: the four
nibbles of a qs word are the four unsigned int8 weights of a B-fragment
register (no table, no signs), one mma.m16n8k32.s8 per sub-block, and the
min term of the format - d sc q - dmin m - taken out of the epilogue with
the activation group's sum: acc += dot * (sx * d sc) - sumx * (dmin m),
where sumx is the sum of the dequantised activations of the group (the
quantiser's optional third output).

The group's 8 qs words (shared by two sub-blocks: low and high nibbles)
are loaded straight into the decode layout [64 rows, 8 words] from the
shared-memory tile (the slice starts on the 8-word swizzle tile), so the
word b of the fragment is the word b of the group, no selects. The scale
and min bytes per sub-block as in the bf16 kernel (get_scale_min_k4),
read in the accumulator's column layout.
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
from .iq3s_int8 import _rowword_in, _word_to_i8x4, quantize_activations

BLOCK_BYTES = 144


@g.jit
def q4k_int8_kernel(XQ, SX, SUMX, W, Y,
                    M, N, num_k_blocks, row_bytes, stride_xq, stride_sx, stride_ym, stride_yk, w_end,
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
    ch: gl.constexpr = BlockedLayout([1, 1], [8, 4], [4, 1], [1, 0])
    cn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, cq))
    cw = gl.arange(0, 32, layout=SliceLayout(0, cq))
    hn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, ch))
    hw = gl.arange(0, 4, layout=SliceLayout(0, ch))
    crow = cn.to(gl.int64) * row_bytes
    hrow = hn.to(gl.int64) * row_bytes
    smem_q = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_qs)
    smem_h = gl.allocate_shared_memory(gl.int32, [2, 64, 4], smem_hd)

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
    src_q0 = (crow + kb0 * 144 + 16)[:, None] + 4 * cw[None, :].to(gl.int64)
    src_h0 = (hrow + kb0 * 144)[:, None] + 4 * hw[None, :].to(gl.int64)
    mq0 = (cn < N)[:, None] & (src_q0 + 4 <= w_end)
    mh0 = (hn < N)[:, None] & (src_h0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem_q.index(0), (W + src_q0).to(gl.pointer_type(gl.int32), bitcast=True), mq0)
    async_copy.async_copy_global_to_shared(smem_h.index(0), (W + src_h0).to(gl.pointer_type(gl.int32), bitcast=True), mh0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            src_q1 = (crow + (kb + 1) * 144 + 16)[:, None] + 4 * cw[None, :].to(gl.int64)
            src_h1 = (hrow + (kb + 1) * 144)[:, None] + 4 * hw[None, :].to(gl.int64)
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
        w0 = _rowword_in(hwords, 0, Lcol_c, Lrow_c)                             # d | dmin << 16
        sw1 = _rowword_in(hwords, 1, Lcol_c, Lrow_c)
        sw2 = _rowword_in(hwords, 2, Lcol_c, Lrow_c)
        sw3 = _rowword_in(hwords, 3, Lcol_c, Lrow_c)
        d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        dmin = ((w0 >> 16) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for gidx in gl.static_range(4):
            qw = qwords.slice(8 * gidx, 8, dim=1).load(Lw)                      # the group's 8 words, [64, 8] in the decode layout
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
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def q4k_int8_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                  quantized: tuple | None = None) -> torch.Tensor:
    """quantized: (int8 X, scales, group sums) from quantize_activations(X, with_sums=True)."""
    BN = 64
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * BLOCK_BYTES and M <= 16
    if W.stride(0) % 4 != 0 or W.data_ptr() % 4 != 0:
        W = W.contiguous()
    if quantized is not None and len(quantized) == 3:
        XQ, SX, SUMX = quantized
    else:
        XQ, SX, SUMX = quantize_activations(X.contiguous(), with_sums=True)
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=BLOCK_BYTES)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    w_end = W.untyped_storage().nbytes() - W.storage_offset()
    q4k_int8_kernel[(triton.cdiv(n_out, BN), splitk)](
        XQ, SX, SUMX, W, Y, M, n_out, nb, W.stride(0), XQ.stride(0), SX.stride(0), Y.stride(1), Y.stride(0), w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
