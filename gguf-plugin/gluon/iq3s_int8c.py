"""IQ3_S x bf16 GEMM in Gluon, the int8 form's fourth variant: the third form with the two loads the SASS
of the third form showed to be the remaining decode budget moved:

  * the A operand: the third form loads 16 x 32 int8 per sub-block by
    predicated 4-byte global loads with a zeroing per register (about 20
    instructions per sub-block with their predicates). Here the k-block's
    A tile (16 x 256 int8, 4 KB) is staged once per k-block by cp.async into
    shared memory (double-buffered with the weight tile, 8 cp.async per
    thread per k-block), and each sub-block's fragment is one shared-memory
    load into the dot-operand layout (the swizzle tile is the 32-byte
    sub-block, as for the weight words);
  * the grid: the third form gathers each lattice word from the 2 KB table
    in global memory (L1-resident) with a 64-bit address pair per gather.
    Here the table sits in shared memory (allocated first, at offset
    TAB_OFF of the dynamic shared memory; the sixth bf16 form's mechanism,
    +5-8 % there) and the gather is one `ld.shared.b32` with a 32-bit
    address, off the L1 latency at 2.75 warps per scheduler.

TAB_OFF is a launcher argument (the allocator's placement of the table is
read off by the test's offset sweep against the third form: the results
are bit-identical when the offset is right and garbage otherwise); the
Gluon-visible read of the table after the loop keeps it allocated.
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
# The grid table's offset in the kernel's dynamic shared memory as the compiler
# lays the allocations out (the weight stage 16 KB, the A tile 8 KB, the table
# after them): found by a sweep, the only offset that reproduces the third form
# bit for bit; a wrong one is garbage that the perplexity identity catches.
TAB_OFF = 24576

# the two shared-memory helpers of the sixth bf16 form, inlined
# (the fifth bf16 form is iq3s.py, which has neither).
@g.jit
def _smem_base(like):
    """The 32-bit shared-space address of the dynamic shared memory (the base
    of every Gluon allocation in this kernel), as a per-element tensor."""
    return gl.inline_asm_elementwise("mov.u32 $0, global_smem;", "=r,r", [like], dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _lds(addr):
    return gl.inline_asm_elementwise("ld.shared.b32 $0, [$1];", "=r,r", [addr], dtype=gl.int32, is_pure=True, pack=1)



@g.jit
def _decode_half_s(qsw, qh, sgw, selq, shq, shs, zero4, tbase):
    qs_4 = zero4 + qsw[:, None]
    qh_4 = zero4 + qh[:, None]
    sg_4 = zero4 + sgw[:, None]
    qsb = _prmt(qs_4, zero4, selq)
    idx = qsb | ((qh_4 << shq) & 0x100)
    gw = _lds(tbase + idx * 4)                              # the magnitude word from the shared table
    nib = (sg_4 >> shs) & 0xF
    x = (nib * 0x204081) & 0x01010101
    return (gw ^ (x * 0xFF)) + x


@g.jit
def iq3s_int8c_kernel(XQ, SX, W, Y, GRID32,
                      M, N, num_k_blocks, row_bytes, stride_xq, stride_sx, stride_ym, stride_yk, w_shift, w_end,
                      BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr, TAB_OFF: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    smem_a8: gl.constexpr = SwizzledSharedLayout(4, 1, 8, [1, 0])       # the same bytes as smem_words, seen as int8
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

    # the grid table first, then the weight tiles, then the activation tiles
    gtab = gl.allocate_shared_memory(gl.int32, [512], tab_layout)
    tl1: gl.constexpr = BlockedLayout([4], [32], [4], [0])
    ti = gl.arange(0, 512, layout=tl1)
    gtab.store(gl.load(GRID32 + ti))
    smem = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_words)
    smem_a = gl.allocate_shared_memory(gl.int32, [2, BM, 64], smem_words)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    cp: gl.constexpr = BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    cn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, cp))
    cw = gl.arange(0, 32, layout=SliceLayout(0, cp))
    crow = cn.to(gl.int64) * row_bytes + w_shift
    # the activation tile: 16 rows x 64 words per k-block, 4-byte cp.async, rows >= M masked
    cpa: gl.constexpr = BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0])
    am = gl.arange(0, BM, layout=SliceLayout(1, cpa))
    aw = gl.arange(0, 64, layout=SliceLayout(0, cpa))
    XQ32 = XQ.to(gl.pointer_type(gl.int32), bitcast=True)
    arow = am.to(gl.int64) * (stride_xq // 4)
    amask = (am < M)[:, None] & (aw < 64)[None, :]

    nrow_w = pid_n * BN + gl.arange(0, BN, layout=Lrow4)
    drow_w = nrow_w.to(gl.int64) * row_bytes + w_shift
    nrow_c = pid_n * BN + gl.arange(0, BN, layout=Lrow_c)
    drow_c = nrow_c.to(gl.int64) * row_bytes + w_shift
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
    o0 = crow + kb0 * 110
    src0 = (o0 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
    m0 = (cn < N)[:, None] & (cw < 28)[None, :] & (src0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem.index(0), (W + src0).to(gl.pointer_type(gl.int32), bitcast=True), m0)
    async_copy.async_copy_global_to_shared(smem_a.index(0), XQ32 + arow[:, None] + (kb0 * 64 + aw)[None, :], amask)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            o1 = crow + (kb + 1) * 110
            src1 = (o1 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
            m1 = (cn < N)[:, None] & (cw < 28)[None, :] & (src1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), (W + src1).to(gl.pointer_type(gl.int32), bitcast=True), m1)
            async_copy.async_copy_global_to_shared(smem_a.index((it + 1) % 2), XQ32 + arow[:, None] + ((kb + 1) * 64 + aw)[None, :], amask)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()   # also orders the table's store before its first read
        words = smem.index(it % 2)
        a_tile = smem_a.index(it % 2)._reinterpret(gl.int8, [BM, 256], smem_a8)
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
            a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)                       # int8 [16, 32] from the staged tile
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            qsw0 = _field_in(words, 2 * ib, selw, Lcol_w, Lrow4)
            qsw1 = _field_in(words, 2 * ib + 1, selw, Lcol_w, Lrow4)
            sgw = _field_in(words, 18 + ib, selw, Lcol_w, Lrow4)
            if ib < 4:
                qh = (qh01 >> (8 * ib)) & 0xFF
            else:
                qh = (qh23 >> (8 * (ib - 4))) & 0xFF
            wq_lo = _decode_half_s(qsw0, qh, sgw, selq, shq_lo, shs_lo, zero4, tbase)
            wq_hi = _decode_half_s(qsw1, qh, sgw, selq, shq_hi, shs_hi, zero4, tbase)
            w8 = gl.join(wq_lo, wq_hi)
            w4 = _word_to_i8x4(gl.join(gl.join(w8, w8), gl.join(w8, w8)))
            b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (2, 1, 3, 4, 0)), [32, 64]), db8)
            acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
            s = (scw >> (4 * ib)) & 0xF
            dl = d * (1.0 + 2.0 * s.to(gl.float32))
            acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    # a Gluon-visible read of the table after the loop keeps it allocated (the
    # inline-asm reads are invisible to the liveness analysis)
    keep = gtab.slice(0, 64).load(SliceLayout(0, mma))
    omask = omask & (keep[None, :] != 0x7FFFFFFF)
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def iq3s_int8c_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                    quantized: tuple | None = None, tab_off: int = TAB_OFF) -> torch.Tensor:
    BN = 64
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * BLOCK_BYTES and M <= 16
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations(X.contiguous())
    assert XQ.stride(0) % 4 == 0 and XQ.data_ptr() % 4 == 0
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=BLOCK_BYTES)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    off = W.storage_offset()
    base = torch.as_strided(W, (1,), (1,), off & -4)
    w_shift = off & 3
    w_end = W.untyped_storage().nbytes() - (off & -4)
    iq3s_int8c_kernel[(triton.cdiv(n_out, BN), splitk)](
        XQ, SX, base, Y, grid32(W.device), M, n_out, nb, W.stride(0), XQ.stride(0), SX.stride(0), Y.stride(1), Y.stride(0), w_shift, w_end,
        BM=16, BN=BN, SPLITK=splitk, TAB_OFF=tab_off, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
