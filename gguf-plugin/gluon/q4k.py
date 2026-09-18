"""Q4_K x bf16 GEMM in Gluon for the decode regime (M <= 16), on the plugin's
weights as they are - the fourth type (13.5 % of the file's bytes, the head
among them), the iq3s_gluon5 scaffold with the Q4_K block.

Q4_K block (256 weights, 144 bytes = 36 words, word-aligned): d f16 | dmin
f16 | scales u8[12] | qs u8[128]. Sub-blocks 2 g and 2 g + 1 share the 32
qs bytes of group g (words 8 g .. 8 g + 7): weight k of sub-block 2 g is the
low nibble of byte k, of sub-block 2 g + 1 the high nibble; the weight is
d * sc(ib) * q - dmin * m(ib) with the 6-bit scale and min of sub-block ib
packed in the 12 scale bytes (llama.cpp's get_scale_min_k4).

The B-operand pair (k, k + 1), k = 4 b + 2 half, is bytes 2 half and
2 half + 1 of word b of the group: the eight words of a group are loaded
once per two sub-blocks straight into the decode layout's (row, word)
slice, one prmt places the pair's bytes at bytes 0 and 2, a shift picks
the nibble, lop3 + fma.bf16x2 make the two 4-bit values exact bf16
(0x4300 | q = 128 + q, minus 128), and one fma.bf16x2 applies scale and min
(q * dl - ml, rounded once). No table, no gather, no straddling (the blocks
are 4-byte aligned): the tile is two word arrays, the 32 qs words and the 4
header words per row, by 4-byte cp.async, double-buffered.
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

BLOCK_BYTES = 144


@g.jit
def _prmt(lo, hi, sel):
    return gl.inline_asm_elementwise("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", [lo, hi, sel],
                                     dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _q4k_pair_asm(nib2, dl2, negml2):
    """pack=2 over the (e) dim: $2 the two nibbles at bits 0..3 and 16..19,
    $4 the bf16x2 scale d * sc, $6 the bf16x2 -dmin * m -> the fragment
    register $0 = q * dl - ml."""
    return gl.inline_asm_elementwise(
        '{\n'
        '    .reg .b32 t, one, neg128;\n'
        '    mov.b32 one, 0x3f803f80;\n'
        '    mov.b32 neg128, 0xc300c300;\n'
        '    lop3.b32 t, $2, 0x000f000f, 0x43004300, 0xea;\n'
        '    fma.rn.bf16x2 t, t, one, neg128;\n'
        '    fma.rn.bf16x2 $0, t, $4, $6;\n'
        '}',
        constraints="=r,r,r,r,r,r,r", args=[nib2, dl2, negml2],
        dtype=gl.bfloat16, is_pure=True, pack=2)


@g.jit
def _rowword(wview, w: gl.constexpr, Lw: gl.constexpr):
    Lcol: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]], lane_bases=[[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 1])
    q = wview.slice(w, 1, dim=1).load(Lcol)
    return gl.convert_layout(gl.reshape(q, [64]), SliceLayout(1, SliceLayout(2, Lw)))


@g.jit
def _to_bf16x2(v):
    """fp32 [BN] -> the bf16 bits duplicated in both halves (int32)."""
    h = v.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    return h | (h << 16)


@g.jit
def q4k_gemm_kernel(X, W, Y,
                    M, N, num_k_blocks, row_bytes, stride_xm, stride_ym, stride_yk, w_end,
                    BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da: gl.constexpr = DotOperandLayout(0, mma, 2)
    db: gl.constexpr = DotOperandLayout(1, mma, 2)
    smem_qs: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])      # [64, 32] words, swizzled by row
    smem_hd: gl.constexpr = SwizzledSharedLayout(1, 1, 1, [1, 0])      # [64, 4] words
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 2, 0], [0, 4, 0], [32, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 1, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
        warp_bases=[[8, 0, 0], [16, 0, 0]], block_bases=[], shape=[64, 8, 2])

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

    nrow = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(1, SliceLayout(2, Lw)))
    half3 = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, Lw)))
    zero3 = gl.zeros([64, 8, 2], gl.int32, Lw)
    h3 = zero3 + half3[None, None, :]
    s3 = 2 * h3
    psel3 = s3 | (s3 << 4) | ((s3 + 1) << 8) | ((s3 + 1) << 12)     # bytes (2 half, 2 half + 1) -> bytes 0, 2
    mm = gl.arange(0, BM, layout=SliceLayout(1, da))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da))
    m_ok = mm < M

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
        qwords = smem_q.index(it % 2)                                       # [64, 32] qs words
        hwords = smem_h.index(it % 2)                                       # [64, 4] header words
        w0 = _rowword(hwords, 0, Lw)                                        # d | dmin << 16
        sw1 = _rowword(hwords, 1, Lw)                                       # scales[0..3]
        sw2 = _rowword(hwords, 2, Lw)                                       # scales[4..7]
        sw3 = _rowword(hwords, 3, Lw)                                       # scales[8..11]
        d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        dmin = ((w0 >> 16) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for gidx in gl.static_range(4):
            # the group's 8 qs words, each lane the words of its (row, b) pairs
            qw = qwords.slice(8 * gidx, 8, dim=1).load(SliceLayout(2, Lw))   # [64, 8]
            qw3 = zero3 + qw[:, :, None]
            t = _prmt(qw3, zero3, psel3)                                    # the pair's bytes at 0 and 2
            for sub in gl.static_range(2):
                ib = 2 * gidx + sub                                          # constexpr arithmetic (static_range values)
                x = gl.load(X + mm[:, None] * stride_xm + (kb * 256 + ib * 32 + kx)[None, :],
                            mask=m_ok[:, None], other=0.0)
                # get_scale_min_k4(ib)
                if ib < 4:
                    sc = (sw1 >> (8 * ib)) & 63
                    mn = (sw2 >> (8 * ib)) & 63
                else:
                    sc = ((sw3 >> (8 * (ib - 4))) & 0xF) | (((sw1 >> (8 * (ib - 4) + 6)) & 3) << 4)
                    mn = ((sw3 >> (8 * (ib - 4) + 4)) & 0xF) | (((sw2 >> (8 * (ib - 4) + 6)) & 3) << 4)
                dl2 = _to_bf16x2(d * sc.to(gl.float32))
                negml2 = _to_bf16x2(-dmin * mn.to(gl.float32))
                dl_3 = zero3 + dl2[:, None, None]
                ml_3 = zero3 + negml2[:, None, None]
                nib2 = t >> (4 * sub)                                        # low or high nibbles (masked in the asm)
                frag = _q4k_pair_asm(gl.join(nib2, nib2), gl.join(dl_3, dl_3), gl.join(ml_3, ml_3))
                wt = gl.reshape(gl.permute(frag, (1, 2, 3, 0)), [32, 64])
                b = gl.convert_layout(wt, db)
                acc = mma_v2(x, b, acc)
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    # stored in the output's dtype: bf16 straight from the kernel without split-K
    # (no cast kernel), fp32 partials otherwise (summed with the cast by the launcher)
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


from .splitk import split_k_for  # noqa: E402  (0033: the split-K against the wave count, one table for every kernel)


def q4k_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """W: the raw Q4_K rows, uint8 [n_out, K/256*144], any row stride with
    4-byte-aligned rows; X bf16 [M <= 16, K]."""
    BN = 64
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * BLOCK_BYTES and M <= 16
    if W.stride(0) % 4 != 0 or W.data_ptr() % 4 != 0:
        # the word-aligned staging needs 4-byte-aligned rows; a shard view of a
        # layer padded to an odd width (no such layer in this model) is copied
        W = W.contiguous()
    X = X.contiguous()
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=BLOCK_BYTES)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    w_end = W.untyped_storage().nbytes() - W.storage_offset()
    q4k_gemm_kernel[(triton.cdiv(n_out, BN), splitk)](
        X, W, Y, M, n_out, nb, W.stride(0), X.stride(0), Y.stride(1), Y.stride(0), w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
