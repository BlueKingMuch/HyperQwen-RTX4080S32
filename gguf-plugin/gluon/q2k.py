"""Q2_K x bf16 GEMM in Gluon for the decode regime (M <= 16), on the plugin's
weights as they are - the last of the eight matmul types (2.0 % of the
file's bytes), the q4k_gluon scaffold with the Q2_K block.

Q2_K block (256 weights, 84 bytes = 21 words, word-aligned): scales u8[16]
(scale nibble | min nibble, one byte per 16 weights) | qs u8[64] (2-bit
quants, four per byte) | d f16 | dmin f16. Weight w = 128 n + 32 t + l
(n = 0, 1; t = 0..3; l = 0..31) is bits 2 t, 2 t + 1 of byte 32 n + l with
scale byte 8 n + 2 t + l // 16; the weight is d sc q - dmin m. In
sub-blocks of 32 (ib = 4 n + t): the 32 bytes of half n hold all four
sub-blocks of that half (one 2-bit field each), and the two 16-weight
halves of a sub-block have their own scale bytes.

The B-operand pair (k, k + 1), k = 4 b + 2 half: bytes 2 half, 2 half + 1
of word b of the half's 8 qs words (loaded once per four sub-blocks
straight into the decode layout's (row, word) slice), one prmt, a shift
by 2 t, lop3 with 0x00030003 | 0x43004300 and fma.bf16x2 make the two
2-bit values exact, one fma applies scale and min (selected by b < 4 for
the 16-weight half). No table, no gather, no straddling; the tile is one
[64, 32]-word array by 4-byte cp.async, double-buffered.
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

BLOCK_BYTES = 84


@g.jit
def _prmt(lo, hi, sel):
    return gl.inline_asm_elementwise("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", [lo, hi, sel],
                                     dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _q2k_pair_asm(q2, dl2, negml2):
    """pack=2 over the (e) dim: $2 the two 2-bit values at bits 0..1 and
    16..17, $4 the bf16x2 scale d sc, $6 the bf16x2 -dmin m -> $0 = q dl - ml."""
    return gl.inline_asm_elementwise(
        '{\n'
        '    .reg .b32 t, one, neg128;\n'
        '    mov.b32 one, 0x3f803f80;\n'
        '    mov.b32 neg128, 0xc300c300;\n'
        '    lop3.b32 t, $2, 0x00030003, 0x43004300, 0xea;\n'
        '    fma.rn.bf16x2 t, t, one, neg128;\n'
        '    fma.rn.bf16x2 $0, t, $4, $6;\n'
        '}',
        constraints="=r,r,r,r,r,r,r", args=[q2, dl2, negml2],
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
    h = v.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    return h | (h << 16)


@g.jit
def q2k_gemm_kernel(X, W, Y,
                    M, N, num_k_blocks, row_bytes, stride_xm, stride_ym, stride_yk, w_end,
                    BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da: gl.constexpr = DotOperandLayout(0, mma, 2)
    db: gl.constexpr = DotOperandLayout(1, mma, 2)
    smem_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 2, 0], [0, 4, 0], [32, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 1, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
        warp_bases=[[8, 0, 0], [16, 0, 0]], block_bases=[], shape=[64, 8, 2])

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    cp: gl.constexpr = BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    cn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, cp))
    cw = gl.arange(0, 32, layout=SliceLayout(0, cp))
    crow = cn.to(gl.int64) * row_bytes
    smem = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_words)

    half3 = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, Lw)))
    bidx3 = gl.arange(0, 8, layout=SliceLayout(0, SliceLayout(2, Lw)))
    zero3 = gl.zeros([64, 8, 2], gl.int32, Lw)
    h3 = zero3 + half3[None, None, :]
    b3 = zero3 + bidx3[None, :, None]
    s3 = 2 * h3
    psel3 = s3 | (s3 << 4) | ((s3 + 1) << 8) | ((s3 + 1) << 12)
    mm = gl.arange(0, BM, layout=SliceLayout(1, da))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da))
    m_ok = mm < M

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    # the tile starts 4 words before the block so that the qs words sit at
    # columns 8..23 (a shared-memory slice must start on the 8-word swizzle tile):
    # columns 0..3 unused (masked: they may fall before the tensor), scales 4..7, dm 24
    src0 = (crow + kb0 * 84 - 16)[:, None] + 4 * cw[None, :].to(gl.int64)
    m0 = (cn < N)[:, None] & (cw < 25)[None, :] & (src0 >= 0) & (src0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem.index(0), (W + src0).to(gl.pointer_type(gl.int32), bitcast=True), m0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            src1 = (crow + (kb + 1) * 84 - 16)[:, None] + 4 * cw[None, :].to(gl.int64)
            m1 = (cn < N)[:, None] & (cw < 25)[None, :] & (src1 >= 0) & (src1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), (W + src1).to(gl.pointer_type(gl.int32), bitcast=True), m1)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        words = smem.index(it % 2)                                          # [64, 32] words: block word j at column j + 4
        dmw = _rowword(words, 24, Lw)                                       # d | dmin << 16 (block word 20 at column 24)
        d = (dmw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        dmin = ((dmw >> 16) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for n in gl.static_range(2):
            qw = words.slice(8 + 8 * n, 8, dim=1).load(SliceLayout(2, Lw))    # the half's 8 qs words (block words 4 + 8 n), [64, 8]
            qw3 = zero3 + qw[:, :, None]
            t = _prmt(qw3, zero3, psel3)                                    # the pair's bytes at 0 and 2
            for tq in gl.static_range(4):
                ib = 4 * n + tq
                x = gl.load(X + mm[:, None] * stride_xm + (kb * 256 + ib * 32 + kx)[None, :],
                            mask=m_ok[:, None], other=0.0)
                # scale bytes is0 = 8 n + 2 tq (weights 0..15) and is0 + 1 (16..31): word (8 n + 2 tq) // 4
                sw = _rowword(words, 4 + 2 * n + tq // 2, Lw)
                sc0 = (sw >> (8 * ((2 * tq) % 4))) & 0xFF
                sc1 = (sw >> (8 * ((2 * tq) % 4 + 1))) & 0xFF
                dl0 = _to_bf16x2(d * (sc0 & 0xF).to(gl.float32))
                dl1 = _to_bf16x2(d * (sc1 & 0xF).to(gl.float32))
                ml0 = _to_bf16x2(-dmin * (sc0 >> 4).to(gl.float32))
                ml1 = _to_bf16x2(-dmin * (sc1 >> 4).to(gl.float32))
                dl_3 = gl.where(b3 < 4, zero3 + dl0[:, None, None], zero3 + dl1[:, None, None])
                ml_3 = gl.where(b3 < 4, zero3 + ml0[:, None, None], zero3 + ml1[:, None, None])
                q2 = t >> (2 * tq)                                          # the 2-bit field (masked in the asm)
                frag = _q2k_pair_asm(gl.join(q2, q2), gl.join(dl_3, dl_3), gl.join(ml_3, ml_3))
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


def q2k_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """W: the raw Q2_K rows, uint8 [n_out, K/256*84], any 4-byte-aligned row stride; X bf16 [M <= 16, K]."""
    BN = 64
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * BLOCK_BYTES and M <= 16
    if W.stride(0) % 4 != 0 or W.data_ptr() % 4 != 0:
        W = W.contiguous()
    X = X.contiguous()
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=BLOCK_BYTES)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    w_end = W.untyped_storage().nbytes() - W.storage_offset()
    q2k_gemm_kernel[(triton.cdiv(n_out, BN), splitk)](
        X, W, Y, M, n_out, nb, W.stride(0), X.stride(0), Y.stride(1), Y.stride(0), w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
