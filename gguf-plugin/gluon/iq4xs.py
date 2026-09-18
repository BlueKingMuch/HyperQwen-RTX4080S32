"""IQ4_XS x bf16 GEMM in Gluon for the decode regime (M <= 16), on the
plugin's weights as they are - the second type after IQ3_S (25.7 % of the
file's bytes), built on the iq3s_gluon5 scaffold.

IQ4_XS block (256 weights, 136 bytes = 34 words, word-aligned): d f16 |
scales_h u16 | scales_l u8[4] | qs u8[128]. Sub-block ib (32 weights) has
qs bytes 16 ib .. 16 ib + 15: weight j < 16 is the low nibble of byte j,
weight 16 + j the high nibble; the nibble indexes llama.cpp's 16-entry
kvalues_iq4nl (-127 .. 113, MIT); the sub-block scale is d * (ls - 32)
with ls the 6-bit scale from scales_l's nibble ib and scales_h's bits
2 ib, 2 ib + 1.

The B-operand pair (k, k + 1), k = 4 b + 2 half, sits in qs word 4 ib +
(b mod 4), bytes 2 half and 2 half + 1, low nibbles for b < 4 and high for
b >= 4: one prmt places the two bytes at bytes 0 and 2, a shift and mask
give the two nibbles, and a 256-entry table of bf16x2 pairs (1 KB, the
signed values of both nibbles) turns them into the exact pair register
with one gather; one fma.bf16x2 applies the scale. No sign nibble, no
straddling: the blocks are 4-byte aligned (136 = 34 words), so the tile is
two word arrays - the 32 qs words and the 2 header words per row - copied
by 4-byte cp.async, double-buffered.
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

BLOCK_BYTES = 136
KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


@g.jit
def _prmt(lo, hi, sel):
    return gl.inline_asm_elementwise("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", [lo, hi, sel],
                                     dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _scale_asm(pair, scale2):
    """pack=2 over the (e) dim: the bf16x2 pair $2 times the bf16x2 scale $4
    -> the fragment register $0 (fma.rn.bf16x2 with zero; mul.bf16x2 needs
    sm_90)."""
    return gl.inline_asm_elementwise(
        '{\n'
        '    .reg .b32 zero;\n'
        '    mov.b32 zero, 0x00000000;\n'
        '    fma.rn.bf16x2 $0, $2, $4, zero;\n'
        '}',
        constraints="=r,r,r,r,r", args=[pair, scale2],
        dtype=gl.bfloat16, is_pure=True, pack=2)


@g.jit
def _rowword(wview, w: gl.constexpr, Lw: gl.constexpr):
    """Word column w of a [64, C]-word tile as a [64] tensor in the row layout
    of the decode (each lane its own rows, the four lanes of a row group
    alike)."""
    Lcol: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]], lane_bases=[[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 1])
    q = wview.slice(w, 1, dim=1).load(Lcol)
    return gl.convert_layout(gl.reshape(q, [64]), SliceLayout(1, SliceLayout(2, Lw)))


@g.jit
def iq4xs_gemm_kernel(X, W, Y, PAIRS,
                      M, N, num_k_blocks, row_bytes, stride_xm, stride_ym, stride_yk, w_end,
                      BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da: gl.constexpr = DotOperandLayout(0, mma, 2)
    db: gl.constexpr = DotOperandLayout(1, mma, 2)
    smem_qs: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])      # [64, 32] words, swizzled by row
    smem_hd: gl.constexpr = SwizzledSharedLayout(1, 1, 1, [1, 0])      # [64, 2] words
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 2, 0], [0, 4, 0], [32, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 1, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
        warp_bases=[[8, 0, 0], [16, 0, 0]], block_bases=[], shape=[64, 8, 2])

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    # copy layouts: the qs words (a warp along a row's 32 words) and the two
    # header words (64 rows x 2 words, one word per thread)
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

    nrow = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(1, SliceLayout(2, Lw)))
    bidx3 = gl.arange(0, 8, layout=SliceLayout(0, SliceLayout(2, Lw)))
    half3 = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, Lw)))
    zero3 = gl.zeros([64, 8, 2], gl.int32, Lw)
    b3 = zero3 + bidx3[None, :, None]
    h3 = zero3 + half3[None, None, :]
    # prmt selector: bytes (s, s + 1) of the word pair (w_{b&1}, w_{b&1 + 1}) to
    # bytes 0 and 2, s = 4 (b & 1) + 2 half
    s3 = 4 * (b3 & 1) + 2 * h3
    psel3 = s3 | (s3 << 4) | ((s3 + 1) << 8) | ((s3 + 1) << 12)
    nsh3 = gl.where(b3 < 4, 0, 4)                                        # low or high nibbles
    mm = gl.arange(0, BM, layout=SliceLayout(1, da))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da))
    m_ok = mm < M

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
        qwords = smem_q.index(it % 2)                                       # [64, 32] qs words
        hwords = smem_h.index(it % 2)                                       # [64, 2] header words
        w0 = _rowword(hwords, 0, Lw)                                        # d | scales_h << 16
        slw = _rowword(hwords, 1, Lw)                                       # scales_l bytes
        d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        shw = (w0 >> 16) & 0xFFFF
        for ib in gl.static_range(8):
            x = gl.load(X + mm[:, None] * stride_xm + (kb * 256 + ib * 32 + kx)[None, :],
                        mask=m_ok[:, None], other=0.0)
            q0 = _rowword(qwords, 4 * ib, Lw)
            q1 = _rowword(qwords, 4 * ib + 1, Lw)
            q2 = _rowword(qwords, 4 * ib + 2, Lw)
            q3 = _rowword(qwords, 4 * ib + 3, Lw)
            ls = ((slw >> (4 * ib)) & 0xF) | (((shw >> (2 * ib)) & 3) << 4)
            dl16 = (d * (ls.to(gl.float32) - 32.0)).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
            scale2 = dl16 | (dl16 << 16)
            q0_3 = zero3 + q0[:, None, None]
            q1_3 = zero3 + q1[:, None, None]
            q2_3 = zero3 + q2[:, None, None]
            q3_3 = zero3 + q3[:, None, None]
            sc_3 = zero3 + scale2[:, None, None]
            # the byte pair of this lane's k: word b & 3 (b < 4 low nibbles, b >= 4 high)
            t01 = _prmt(q0_3, q1_3, psel3)
            t23 = _prmt(q2_3, q3_3, psel3)
            t = gl.where((b3 & 2) == 0, t01, t23) >> nsh3
            idx = (t & 0xF) | (((t >> 16) & 0xF) << 4)
            pair = gl.load(PAIRS + idx)                                      # bf16x2 of the two signed values
            frag = _scale_asm(gl.join(pair, pair), gl.join(sc_3, sc_3))      # [BN, 8, 2, 2] bf16
            wt = gl.reshape(gl.permute(frag, (1, 2, 3, 0)), [32, 64])
            b = gl.convert_layout(wt, db)
            acc = mma_v2(x, b, acc)
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    # this split's partial sum to its own slice of Y (deterministic: the
    # slices are summed in a fixed order by the launcher; fp32 atomics made
    # the result vary between runs)
    Yp = Y + pid_k.to(gl.int64) * stride_yk
    # stored in the output's dtype: bf16 straight from the kernel without split-K
    # (no cast kernel), fp32 partials otherwise (summed with the cast by the launcher)
    gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


_PAIRS: dict = {}


def pairs_table(device: torch.device) -> torch.Tensor:
    """[256] int32: entry (n0 | n1 << 4) = bf16x2 (kvalues[n0], kvalues[n1])."""
    key = (device.type, device.index)
    if key not in _PAIRS:
        try:
            from gguf.quants import IQ4_NL
            kv = tuple(int(v) for v in IQ4_NL.kvalues)
        except Exception:  # noqa: BLE001
            kv = KVALUES
        assert kv == KVALUES, kv
        vals = torch.tensor(kv, dtype=torch.float32)
        lo = vals.repeat(16)                       # n0 fastest
        hi = vals.repeat_interleave(16)
        bl = lo.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        bh = hi.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        _PAIRS[key] = (bl | (bh << 16)).to(torch.int32).contiguous().to(device)
    return _PAIRS[key]


from .splitk import split_k_for  # noqa: E402  (0033: the split-K against the wave count, one table for every kernel)


def iq4xs_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """W: the raw IQ4_XS rows, uint8 [n_out, K/256*136], any row stride with
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
    iq4xs_gemm_kernel[(triton.cdiv(n_out, BN), splitk)](
        X, W, Y, pairs_table(W.device), M, n_out, nb, W.stride(0), X.stride(0), Y.stride(1), Y.stride(0), w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
