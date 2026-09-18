"""IQ3_S x bf16 GEMM in Gluon for the decode regime (M <= 16), reading the
plugin's weights as they are.

    Y[M, N] = X[M, K] . W[N, K]^T,   W the raw IQ3_S rows (uint8 [N, K/256*110],
                                     any row stride, e.g. a merged layer's shard view)

Per 32-weight sub-block of a row: eight qs bytes plus the qh bit give eight
9-bit grid indices; each grid entry is one 32-bit word of four magnitude
bytes; the word's byte pair of a lane goes through prmt / lop3 / fma into an
exact bf16x2 register, the sign nibble through an xor on the bf16 sign bits,
the sub-block scale d * (1 + 2 s) through one fma.bf16x2; the registers are
the B operand of mma_v2 directly (the decode layout is built on the
measured bases of the operand). The tile of 64 rows x 110 bytes is staged
by 4-byte cp.async chunks (double-buffered) from the 4-byte-aligned address
at or 2 bytes before each block, every field taken out of the two aligned
words it may straddle by one prmt; the copy is masked against the end of
the storage. Split-K with a deterministic fp32 epilogue (partials summed in a fixed order), chosen from the shape.

Measured on this recipe's RTX 4080 SUPER under CUDA graphs: attn_qkv
[10240x5120] 0.057 ms (395 GB/s equivalent) and ffn_down [5120x17408]
0.093 ms (411 GB/s) at 8 rows, 3.1x the batched MMVQ; exact against
gguf.quants.dequantize (max relative error 3-5e-3, the bf16 rounding of
the products).
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


_GRID: dict = {}


def grid32(device: torch.device) -> torch.Tensor:
    """[512] int32: the IQ3_S grid (llama.cpp's iq3s_grid, MIT; taken from the
    gguf package's tables) as words of four magnitude bytes (1..15)."""
    key = (device.type, device.index)
    if key not in _GRID:
        import numpy as np
        from gguf.quants import IQ3_S

        IQ3_S.init_grid()
        g8 = IQ3_S.grid[0, 0].astype(np.uint8)                                      # [512, 4]
        _GRID[key] = torch.from_numpy(g8.copy()).view(torch.int32).flatten().to(device)
    return _GRID[key]

BLOCK_BYTES = 110


@g.jit
def _prmt(lo, hi, sel):
    """prmt.b32 $0, lo, hi, sel: four bytes picked out of the 8-byte pair."""
    return gl.inline_asm_elementwise("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", [lo, hi, sel],
                                     dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _pair_asm(word, sel, smask, scale2):
    """pack=2 over the (e) dim: the grid word $2, prmt selector $4, sign xor
    mask $6, bf16x2 scale $8 -> the fragment register $0."""
    return gl.inline_asm_elementwise(
        '{\n'
        '    .reg .b32 t, one, neg128, zero;\n'
        '    mov.b32 one, 0x3f803f80;\n'
        '    mov.b32 neg128, 0xc300c300;\n'
        '    mov.b32 zero, 0x00000000;\n'
        '    prmt.b32 t, $2, zero, $4;\n'
        '    lop3.b32 t, t, 0x000f000f, 0x43004300, 0xea;\n'
        '    fma.rn.bf16x2 t, t, one, neg128;\n'
        '    xor.b32 t, t, $6;\n'
        '    fma.rn.bf16x2 $0, t, $8, zero;\n'
        '}',
        constraints="=r,r,r,r,r,r,r,r,r", args=[word, sel, smask, scale2],
        dtype=gl.bfloat16, is_pure=True, pack=2)


@g.jit
def _rowword(wview, w: gl.constexpr, Lw: gl.constexpr):
    """Aligned word column w of the [64, 32]-word tile as a [64] tensor in the
    row layout of the decode (each lane its own rows, the four lanes of a row
    group alike)."""
    Lcol: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]], lane_bases=[[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 1])
    q = wview.slice(w, 1, dim=1).load(Lcol)
    return gl.convert_layout(gl.reshape(q, [64]), SliceLayout(1, SliceLayout(2, Lw)))


@g.jit
def _field(wview, w: gl.constexpr, sel, Lw: gl.constexpr):
    """The 4-byte field at block byte j = 4 w + 2: prmt of aligned words w and
    w + 1 with the row's selector (0x5432 when the block starts on a word,
    0x7654 when it starts 2 bytes in)."""
    return _prmt(_rowword(wview, w, Lw), _rowword(wview, w + 1, Lw), sel)


@g.jit
def iq3s_gemm_kernel(X, W, Y, GRID32,
                      M, N, num_k_blocks, row_bytes, stride_xm, stride_ym, stride_yk, w_shift, w_end,
                      BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da: gl.constexpr = DotOperandLayout(0, mma, 2)
    db: gl.constexpr = DotOperandLayout(1, mma, 2)
    # word tile [64 rows, 32 words] (28 used), words XOR-swizzled by the row so
    # a column read of the eight row groups of a warp hits eight banks
    smem_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
    # decode layout over [BN rows, 8 words, 2 halves] on the measured bases of
    # the B operand (gluon_db_layout_probe.py): registers word bits 1, 2 and
    # row 32; lanes half, word bit 0, rows 1, 2, 4; warps rows 8, 16
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 2, 0], [0, 4, 0], [32, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 1, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
        warp_bases=[[8, 0, 0], [16, 0, 0]], block_bases=[], shape=[64, 8, 2])

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    # copy layout: one word per thread, a warp along a row's 32 words
    cp: gl.constexpr = BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    cn = pid_n * BN + gl.arange(0, 64, layout=SliceLayout(1, cp))
    cw = gl.arange(0, 32, layout=SliceLayout(0, cp))
    crow = cn.to(gl.int64) * row_bytes + w_shift                           # [64] byte offset of the row from the 4-byte-aligned base
    smem = gl.allocate_shared_memory(gl.int32, [2, 64, 32], smem_words)

    # index tensors of the decode layout
    nrow = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(1, SliceLayout(2, Lw)))                       # [BN]
    drow = nrow.to(gl.int64) * row_bytes + w_shift
    bidx3 = gl.arange(0, 8, layout=SliceLayout(0, SliceLayout(2, Lw)))     # [8] word index b (per lane: its parity)
    half3 = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, Lw)))     # [2] half (per lane: bit 0)
    zero3 = gl.zeros([64, 8, 2], gl.int32, Lw)
    b3 = zero3 + bidx3[None, :, None]
    h3 = zero3 + half3[None, None, :]
    sel3 = gl.where(h3 == 0, 0x4140, 0x4342)
    # A operand indices
    mm = gl.arange(0, BM, layout=SliceLayout(1, da))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da))
    m_ok = mm < M

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    # prologue: tile 0. Block start o = row + kb * 110 (2-byte aligned); the
    # copy starts at o & -4 and takes 28 words; chunks past the storage end
    # or of rows past N are masked (the mask is per word, cp.async of 4 bytes)
    o0 = crow + kb0 * 110
    src0 = (o0 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
    m0 = (cn < N)[:, None] & (cw < 28)[None, :] & (src0 + 4 <= w_end)
    async_copy.async_copy_global_to_shared(smem.index(0), (W + src0).to(gl.pointer_type(gl.int32), bitcast=True), m0)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            o1 = crow + (kb + 1) * 110
            src1 = (o1 & -4)[:, None] + 4 * cw[None, :].to(gl.int64)
            m1 = (cn < N)[:, None] & (cw < 28)[None, :] & (src1 + 4 <= w_end)
            async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2), (W + src1).to(gl.pointer_type(gl.int32), bitcast=True), m1)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        words = smem.index(it % 2)                                          # [64, 32] words
        # per row: does the block start 2 bytes into word 0?  (o & 2)
        s2 = ((drow + kb * 110) & 2).to(gl.int32)                           # [BN] 0 or 2
        selb = gl.where(s2 == 0, 0x5432, 0x7654)                            # fields at block byte 4w + 2
        dw = _rowword(words, 0, Lw)
        d = ((dw >> (8 * s2)) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        qh01 = _field(words, 16, selb, Lw)                                  # qh bytes 0..3 (block byte 66)
        qh23 = _field(words, 17, selb, Lw)                                  # qh bytes 4..7 (70)
        scw = _field(words, 26, selb, Lw)                                   # scale nibbles (106)
        for ib in gl.static_range(8):
            x = gl.load(X + mm[:, None] * stride_xm + (kb * 256 + ib * 32 + kx)[None, :],
                        mask=m_ok[:, None], other=0.0)
            qsw0 = _field(words, 2 * ib, selb, Lw)                          # qs bytes 0..3 (block byte 2 + 8 ib)
            qsw1 = _field(words, 2 * ib + 1, selb, Lw)                      # qs bytes 4..7 (6 + 8 ib)
            sgw = _field(words, 18 + ib, selb, Lw)                          # sign bytes (74 + 4 ib)
            if ib < 4:
                qh = (qh01 >> (8 * ib)) & 0xFF
            else:
                qh = (qh23 >> (8 * (ib - 4))) & 0xFF
            s = (scw >> (4 * ib)) & 0xF
            dl16 = (d * (1.0 + 2.0 * s.to(gl.float32))).to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
            scale2 = dl16 | (dl16 << 16)
            qsw0_3 = zero3 + qsw0[:, None, None]
            qsw1_3 = zero3 + qsw1[:, None, None]
            qh_3 = zero3 + qh[:, None, None]
            sg_3 = zero3 + sgw[:, None, None]
            sc_3 = zero3 + scale2[:, None, None]
            qsb = gl.where(b3 < 4, qsw0_3 >> (8 * b3), qsw1_3 >> (8 * (b3 - 4))) & 0xFF
            idx = qsb | (((qh_3 >> b3) & 1) << 8)
            gw = gl.load(GRID32 + idx)                                      # [BN, 8, 2] grid words (halves alike)
            nib = (sg_3 >> (4 * b3)) & 0xF
            sh = 2 * h3
            smask = (((nib >> sh) & 1) << 15) | (((nib >> (sh + 1)) & 1) << 31)
            frag = _pair_asm(gl.join(gw, gw), gl.join(sel3, sel3), gl.join(smask, smask), gl.join(sc_3, sc_3))  # [BN, 8, 2, 2] bf16
            wt = gl.reshape(gl.permute(frag, (1, 2, 3, 0)), [32, 64])
            b = gl.convert_layout(wt, db)
            acc = mma_v2(x, b, acc)
        gl.barrier()   # every warp done with this buffer before it is refilled
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


from .splitk import split_k_for  # noqa: E402  (0033: the split-K against the wave count, one table for every kernel)


def iq3s_gemm(W: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """W: the raw IQ3_S rows, uint8 [n_out, K/256*110], possibly a row-strided
    view (a merged layer's shard); X bf16 [M <= 16, K]."""
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
    # the kernel's byte offsets are taken from a 4-byte-aligned base: the view's
    # storage offset rounded down (a shard view may start 2 bytes into a word)
    off = W.storage_offset()
    base = torch.as_strided(W, (1,), (1,), off & -4)
    w_shift = off & 3
    w_end = W.untyped_storage().nbytes() - (off & -4)
    iq3s_gemm_kernel[(triton.cdiv(n_out, BN), splitk)](
        X, base, Y, grid32(W.device), M, n_out, nb, W.stride(0), X.stride(0), Y.stride(1), Y.stride(0), w_shift, w_end,
        BM=16, BN=BN, SPLITK=splitk, num_warps=num_warps)
    return torch.sum(Y, 0, dtype=X.dtype) if splitk > 1 else Y[0]
