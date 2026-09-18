"""IQ3_XXS x bf16 GEMM in Gluon for the decode regime (M <= 16), on the
plugin's weights as they are - the third type (17.6 % of the file's bytes),
the iq3s_gluon5 scaffold with the IQ3_XXS block.

IQ3_XXS block (256 weights, 98 bytes, 2-byte aligned like IQ3_S): d f16 |
qs u8[64] | u32[8]. Sub-block ib (32 weights): qs bytes 8 ib .. 8 ib + 7 are
8-bit indices into llama.cpp's iq3xxs_grid (256 words of four magnitudes,
MIT; taken from the gguf package's tables); the u32 at 66 + 4 ib holds four
7-bit sign indices (bits 7 l .. 7 l + 6 for the 8 weights of group l, the
8th sign bit of a group the parity of the seven) and the 4-bit scale s in
its top bits; the sub-block scale is d * (2 s + 1) / 4.

The B-operand pair (k, k + 1), k = 4 b + 2 half: grid word b of the
sub-block, bytes 2 half and 2 half + 1, signs from group b // 2 at bits
4 (b mod 2) + 2 half and + 1 - the IQ3_S decode with the 9th grid bit
replaced by a full byte and the sign nibble by the parity-extended 7-bit
index (popc). Same unaligned staging (4-byte cp.async from the aligned
address at or 2 bytes before the block, fields by one prmt per straddle).
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

BLOCK_BYTES = 98


@g.jit
def _prmt(lo, hi, sel):
    return gl.inline_asm_elementwise("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", [lo, hi, sel],
                                     dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _popc(v):
    return gl.inline_asm_elementwise("popc.b32 $0, $1;", "=r,r", [v], dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _pair_asm(word, sel, smask, scale2):
    """pack=2 over the (e) dim: the grid word $2, prmt selector $4, sign xor
    mask $6, bf16x2 scale $8 -> the fragment register $0. The magnitudes are
    7-bit here (4..62), hence the 0x007f007f mask (IQ3_S: 4-bit)."""
    return gl.inline_asm_elementwise(
        '{\n'
        '    .reg .b32 t, one, neg128, zero;\n'
        '    mov.b32 one, 0x3f803f80;\n'
        '    mov.b32 neg128, 0xc300c300;\n'
        '    mov.b32 zero, 0x00000000;\n'
        '    prmt.b32 t, $2, zero, $4;\n'
        '    lop3.b32 t, t, 0x007f007f, 0x43004300, 0xea;\n'
        '    fma.rn.bf16x2 t, t, one, neg128;\n'
        '    xor.b32 t, t, $6;\n'
        '    fma.rn.bf16x2 $0, t, $8, zero;\n'
        '}',
        constraints="=r,r,r,r,r,r,r,r,r", args=[word, sel, smask, scale2],
        dtype=gl.bfloat16, is_pure=True, pack=2)


@g.jit
def _rowword(wview, w: gl.constexpr, Lw: gl.constexpr):
    Lcol: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]], lane_bases=[[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 1])
    q = wview.slice(w, 1, dim=1).load(Lcol)
    return gl.convert_layout(gl.reshape(q, [64]), SliceLayout(1, SliceLayout(2, Lw)))


@g.jit
def _field(wview, w: gl.constexpr, sel, Lw: gl.constexpr):
    """The 4-byte field at block byte 4 w + 2 out of aligned words w, w + 1."""
    return _prmt(_rowword(wview, w, Lw), _rowword(wview, w + 1, Lw), sel)


@g.jit
def iq3xxs_gemm_kernel(X, W, Y, GRID32,
                       M, N, num_k_blocks, row_bytes, stride_xm, stride_ym, stride_yk, w_shift, w_end,
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
    crow = cn.to(gl.int64) * row_bytes + w_shift
    smem = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_words)

    nrow = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(1, SliceLayout(2, Lw)))
    drow = nrow.to(gl.int64) * row_bytes + w_shift
    bidx3 = gl.arange(0, 8, layout=SliceLayout(0, SliceLayout(2, Lw)))
    half3 = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, Lw)))
    zero3 = gl.zeros([64, 8, 2], gl.int32, Lw)
    b3 = zero3 + bidx3[None, :, None]
    h3 = zero3 + half3[None, None, :]
    sel3 = gl.where(h3 == 0, 0x4140, 0x4342)
    gsh3 = 7 * (b3 >> 1)                       # the group's 7 sign bits in the u32
    jsh3 = 4 * (b3 & 1) + 2 * h3               # the pair's bits in the group's 8-bit mask
    mm = gl.arange(0, BM, layout=SliceLayout(1, da))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da))
    m_ok = mm < M

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    o0 = crow + kb0 * 98
    src0 = (o0 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
    m0 = (cn < N)[:, None] & (cw < 25)[None, :] & (src0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem.index(0), (W + src0).to(gl.pointer_type(gl.int32), bitcast=True), m0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            o1 = crow + (kb + 1) * 98
            src1 = (o1 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
            m1 = (cn < N)[:, None] & (cw < 25)[None, :] & (src1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), (W + src1).to(gl.pointer_type(gl.int32), bitcast=True), m1)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        words = smem.index(it % 2)
        s2 = ((drow + kb * 98) & 2).to(gl.int32)
        selb = gl.where(s2 == 0, 0x5432, 0x7654)
        dw = _rowword(words, 0, Lw)
        d = ((dw >> (8 * s2)) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for ib in gl.static_range(8):
            x = gl.load(X + mm[:, None] * stride_xm + (kb * 256 + ib * 32 + kx)[None, :],
                        mask=m_ok[:, None], other=0.0)
            qsw0 = _field(words, 2 * ib, selb, Lw)                          # qs bytes 0..3 (block byte 2 + 8 ib)
            qsw1 = _field(words, 2 * ib + 1, selb, Lw)                      # qs bytes 4..7
            aux = _field(words, 16 + ib, selb, Lw)                          # signs | scale << 28 (66 + 4 ib)
            s = (aux >> 28) & 0xF
            dl16 = (d * (2.0 * s.to(gl.float32) + 1.0) * 0.25).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
            scale2 = dl16 | (dl16 << 16)
            qsw0_3 = zero3 + qsw0[:, None, None]
            qsw1_3 = zero3 + qsw1[:, None, None]
            aux_3 = zero3 + aux[:, None, None]
            sc_3 = zero3 + scale2[:, None, None]
            idx = gl.where(b3 < 4, qsw0_3 >> (8 * b3), qsw1_3 >> (8 * (b3 - 4))) & 0xFF
            gw = gl.load(GRID32 + idx)
            s7 = (aux_3 >> gsh3) & 127
            mask8 = s7 | ((_popc(s7) & 1) << 7)
            smask = (((mask8 >> jsh3) & 1) << 15) | (((mask8 >> (jsh3 + 1)) & 1) << 31)
            frag = _pair_asm(gl.join(gw, gw), gl.join(sel3, sel3), gl.join(smask, smask), gl.join(sc_3, sc_3))
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


_GRID: dict = {}


def grid32(device: torch.device) -> torch.Tensor:
    """[256] int32: the IQ3_XXS grid as words of four magnitude bytes."""
    key = (device.type, device.index)
    if key not in _GRID:
        import numpy as np
        from gguf.quants import IQ3_XXS

        IQ3_XXS.init_grid()
        g8 = IQ3_XXS.grid[0, 0].astype(np.uint8)                                    # [256, 4]
        assert g8.shape == (256, 4) and g8.max() <= 127, g8.shape
        _GRID[key] = torch.from_numpy(g8.copy()).view(torch.int32).flatten().to(device)
    return _GRID[key]


from .splitk import split_k_for  # noqa: E402  (0033: the split-K against the wave count, one table for every kernel)


def iq3xxs_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """W: the raw IQ3_XXS rows, uint8 [n_out, K/256*98], any row stride; X bf16 [M <= 16, K]."""
    BN = 64
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * BLOCK_BYTES and M <= 16
    X = X.contiguous()
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=BLOCK_BYTES)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    off = W.storage_offset()
    base = torch.as_strided(W, (1,), (1,), off & -4)
    w_shift = off & 3
    w_end = W.untyped_storage().nbytes() - (off & -4)
    iq3xxs_gemm_kernel[(triton.cdiv(n_out, BN), splitk)](
        X, base, Y, grid32(W.device), M, n_out, nb, W.stride(0), X.stride(0), Y.stride(1), Y.stride(0), w_shift, w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
