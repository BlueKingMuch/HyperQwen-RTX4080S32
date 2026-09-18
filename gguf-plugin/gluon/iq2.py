"""IQ2_XXS / IQ2_XS / IQ2_S x bf16 GEMM in Gluon for the decode regime
(M <= 16), on the plugin's weights as they are - the three E8-lattice
2-bit types (9.5 % of the file's bytes; 12 ms of the 56 ms decode step on
the plugin's MMVQ, plus the shard copies they cause), one kernel with a
constexpr type switch on the iq3s_gluon5 scaffold.

All three share the group structure: a sub-block of 32 weights is four
groups of 8, each group one entry of an 8-byte lattice table (magnitudes
8, 25, 43, 67 in eighth units; llama.cpp's iq2xxs_grid[256],
iq2xs_grid[512], iq2s_grid[1024], MIT, taken from the gguf package's
tables), one 8-bit sign mask and a scale per 16 weights; the weight is
d (2 s + 1) / 8 x magnitude x sign.

  IQ2_XXS (66 bytes): d | 8 x u16[4] per sub-block: bytes 0..3 the four
    8-bit grid indices, the u32 of the last two u16 four 7-bit sign indices
    (bits 7 g) and the 4-bit scale (bits 28..31), one scale per sub-block.
  IQ2_XS (74 bytes): d | u16[32]: per group a 9-bit grid index (bits 0..8)
    and a 7-bit sign index (bits 9..15) | scales[8]: two nibbles per
    sub-block (groups 0, 1 the low, 2, 3 the high).
  IQ2_S (82 bytes): d | qs[64]: bytes 0..31 the 8-bit grid indices, bytes
    32..63 the sign bytes | qh[8]: bits 2 g, 2 g + 1 of sub-block ib the
    index bits 8, 9 | scales[8] as IQ2_XS.

The 7-bit sign indices are extended by their parity (popc) into the 8-bit
mask instead of llama.cpp's ksigns table. The B-operand pair (k, k + 1),
k = 4 b + 2 half: group b // 2, table word b mod 2, bytes 2 half and
2 half + 1 of it; the IQ3_S pair asm with a 7-bit magnitude mask. The
blocks are 2-byte aligned, so the unaligned staging of iq3s_gluon5
applies (4-byte cp.async from the aligned address at or 2 bytes before
the block, fields by one prmt per straddle).
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

GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ2_S = 22
BLOCK_BYTES = {GGML_TYPE_IQ2_XXS: 66, GGML_TYPE_IQ2_XS: 74, GGML_TYPE_IQ2_S: 82}


@g.jit
def _prmt(lo, hi, sel):
    return gl.inline_asm_elementwise("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", [lo, hi, sel],
                                     dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _popc(v):
    return gl.inline_asm_elementwise("popc.b32 $0, $1;", "=r,r", [v], dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _pair_asm(word, sel, smask, scale2):
    """pack=2 over the (e) dim: the table word $2, prmt selector $4, sign xor
    mask $6, bf16x2 scale $8 -> the fragment register $0 (7-bit magnitudes)."""
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
def iq2_gemm_kernel(X, W, Y, GRID,
                    M, N, num_k_blocks, row_bytes, stride_xm, stride_ym, stride_yk, w_shift, w_end,
                    TYPE: gl.constexpr, BLOCK: gl.constexpr, WORDS: gl.constexpr,
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
    g3 = b3 >> 1                                # group of the pair
    w3 = b3 & 1                                 # table word within the group's 8 bytes
    jsh3 = 4 * w3 + 2 * h3                      # the pair's bits in the group's 8-bit sign mask
    mm = gl.arange(0, BM, layout=SliceLayout(1, da))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da))
    m_ok = mm < M

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
        s2 = ((drow + kb * BLOCK) & 2).to(gl.int32)
        selb = gl.where(s2 == 0, 0x5432, 0x7654)
        dw = _rowword(words, 0, Lw)
        d = ((dw >> (8 * s2)) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for ib in gl.static_range(8):
            x = gl.load(X + mm[:, None] * stride_xm + (kb * 256 + ib * 32 + kx)[None, :],
                        mask=m_ok[:, None], other=0.0)
            if TYPE == 16:      # IQ2_XXS
                f0 = _field(words, 2 * ib, selb, Lw)                    # the four 8-bit grid indices (block byte 2 + 8 ib)
                aux = _field(words, 2 * ib + 1, selb, Lw)               # signs (7 bits per group) | scale << 28
                s = (aux >> 28) & 0xF
                dl16 = (d * (2.0 * s.to(gl.float32) + 1.0) * 0.125).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
                scale2 = dl16 | (dl16 << 16)
                f0_3 = zero3 + f0[:, None, None]
                aux_3 = zero3 + aux[:, None, None]
                sc_3 = zero3 + scale2[:, None, None]
                idx = (f0_3 >> (8 * g3)) & 0xFF
                s7 = (aux_3 >> (7 * g3)) & 127
                mask8 = s7 | ((_popc(s7) & 1) << 7)
            elif TYPE == 17:    # IQ2_XS
                f0 = _field(words, 2 * ib, selb, Lw)                    # u16 of groups 0, 1
                f1 = _field(words, 2 * ib + 1, selb, Lw)                # u16 of groups 2, 3
                scb = (_field(words, 16 + ib // 4, selb, Lw) >> (8 * (ib % 4))) & 0xFF   # scales[ib] (block byte 66 + ib)
                s_lo = scb & 0xF
                s_hi = (scb >> 4) & 0xF
                dl_lo = (d * (2.0 * s_lo.to(gl.float32) + 1.0) * 0.125).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
                dl_hi = (d * (2.0 * s_hi.to(gl.float32) + 1.0) * 0.125).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
                f0_3 = zero3 + f0[:, None, None]
                f1_3 = zero3 + f1[:, None, None]
                lo_3 = zero3 + (dl_lo | (dl_lo << 16))[:, None, None]
                hi_3 = zero3 + (dl_hi | (dl_hi << 16))[:, None, None]
                q2 = gl.where(g3 < 2, f0_3 >> (16 * (g3 & 1)), f1_3 >> (16 * (g3 & 1))) & 0xFFFF
                idx = q2 & 511
                s7 = q2 >> 9
                mask8 = s7 | ((_popc(s7) & 1) << 7)
                sc_3 = gl.where(g3 < 2, lo_3, hi_3)
            else:               # IQ2_S (22)
                f0 = _field(words, ib, selb, Lw)                        # the four 8-bit grid indices (block byte 2 + 4 ib)
                sg = _field(words, 8 + ib, selb, Lw)                    # the four sign bytes (34 + 4 ib)
                qhb = (_field(words, 16 + ib // 4, selb, Lw) >> (8 * (ib % 4))) & 0xFF   # qh[ib] (66 + ib)
                scb = (_field(words, 18 + ib // 4, selb, Lw) >> (8 * (ib % 4))) & 0xFF   # scales[ib] (74 + ib)
                s_lo = scb & 0xF
                s_hi = (scb >> 4) & 0xF
                dl_lo = (d * (2.0 * s_lo.to(gl.float32) + 1.0) * 0.125).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
                dl_hi = (d * (2.0 * s_hi.to(gl.float32) + 1.0) * 0.125).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
                f0_3 = zero3 + f0[:, None, None]
                sg_3 = zero3 + sg[:, None, None]
                qh_3 = zero3 + qhb[:, None, None]
                lo_3 = zero3 + (dl_lo | (dl_lo << 16))[:, None, None]
                hi_3 = zero3 + (dl_hi | (dl_hi << 16))[:, None, None]
                idx = ((f0_3 >> (8 * g3)) & 0xFF) | (((qh_3 >> (2 * g3)) & 3) << 8)
                mask8 = (sg_3 >> (8 * g3)) & 0xFF
                sc_3 = gl.where(g3 < 2, lo_3, hi_3)
            gw = gl.load(GRID + idx * 2 + w3)                            # the pair's table word
            smask = (((mask8 >> jsh3) & 1) << 15) | (((mask8 >> (jsh3 + 1)) & 1) << 31)
            frag = _pair_asm(gl.join(gw, gw), gl.join(sel3, sel3), gl.join(smask, smask), gl.join(sc_3, sc_3))
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


_GRID: dict = {}


def grid_words(quant_type: int, device: torch.device) -> torch.Tensor:
    """The type's lattice table as int32 words: entry i -> words 2 i (bytes
    0..3) and 2 i + 1 (bytes 4..7), magnitudes 8 / 25 / 43 / 67."""
    key = (quant_type, device.type, device.index)
    if key not in _GRID:
        import numpy as np
        from gguf.quants import IQ2_S, IQ2_XS, IQ2_XXS

        cls = {16: IQ2_XXS, 17: IQ2_XS, 22: IQ2_S}[quant_type]
        cls.init_grid()
        g8 = cls.grid[0, 0].astype(np.uint8)
        n = {16: 256, 17: 512, 22: 1024}[quant_type]
        assert g8.shape == (n, 8) and g8.max() <= 127, g8.shape
        _GRID[key] = torch.from_numpy(g8.copy()).view(torch.int32).flatten().to(device)
    return _GRID[key]


from .splitk import split_k_for  # noqa: E402  (0033: the split-K against the wave count, one table for every kernel)


def iq2_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, quant_type: int, splitk: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """W: the raw rows of the type (uint8 [n_out, K/256*bytes], any row stride); X bf16 [M <= 16, K]."""
    BN = 64
    block = BLOCK_BYTES[quant_type]
    assert W.dtype == torch.uint8 and W.is_cuda and W.dim() == 2 and W.stride(1) == 1
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    assert K % 256 == 0 and W.shape[0] == n_out and W.shape[1] == K // 256 * block and M <= 16
    X = X.contiguous()
    nb = K // 256
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=block)
    splitk = max(1, min(splitk, nb))
    Y = torch.empty((splitk, M, n_out), dtype=torch.float32 if splitk > 1 else X.dtype, device=X.device)
    off = W.storage_offset()
    base = torch.as_strided(W, (1,), (1,), off & -4)
    w_shift = off & 3
    w_end = W.untyped_storage().nbytes() - (off & -4)
    iq2_gemm_kernel[(triton.cdiv(n_out, BN), splitk)](
        X, base, Y, grid_words(quant_type, W.device), M, n_out, nb, W.stride(0), X.stride(0), Y.stride(1), Y.stride(0), w_shift, w_end,
        TYPE=quant_type, BLOCK=block, WORDS=(block + 2 + 3) // 4, BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]


def iq2xxs_gemm(W, X, n_out, splitk=None, num_warps=4):
    return iq2_gemm(W, X, n_out, GGML_TYPE_IQ2_XXS, splitk, num_warps)


def iq2xs_gemm(W, X, n_out, splitk=None, num_warps=4):
    return iq2_gemm(W, X, n_out, GGML_TYPE_IQ2_XS, splitk, num_warps)


def iq2s_gemm(W, X, n_out, splitk=None, num_warps=4):
    return iq2_gemm(W, X, n_out, GGML_TYPE_IQ2_S, splitk, num_warps)
