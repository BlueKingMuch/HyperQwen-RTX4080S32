"""IQ3_XXS x bf16 GEMM in Gluon, the int8 form on a tile-major repack with
16-byte copies, in the pattern the other tiled int8 forms use. The 98-byte
block is laid out per row as 25 words: the
64 grid-index bytes (block bytes 2..65) in words 0-15, the eight sign/scale
words (66..97) in words 16-23, d in the low half of word 24 (the high half
zero) - 100 bytes, an odd word count (conflict-free column reads). The rows
are tile-major; a tile's 1,600 words are copied linearly in 16-byte chunks
(one masked chunk of 2,048) into a flat [2, 2048]-word shared array, and
the fields are read as ld.shared gathers at row * 25 + w: the row form's
field k (block byte 2 + 4 k) is tile word k, so the decode (8-bit grid
indices into the 256-word table, the 7-bit sign fields with their parity
bit, d (2 s + 1) / 4), the direct A loads, the mma and the epilogue are the
int8 form's unchanged: bit-identical.

The split-K reduction inside the kernel is an option (reduce=; +2.5 % on C1
in situ, so the sum kernel stays the default), and the output view is out=.
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

from .splitk_reduce import _splitk_reduce, splitk_buffers, splitk_result   # 0039: the in-kernel split-K reduction

from .iq3s import split_k_for
from .iq3s_int8 import _word_to_i8x4, quantize_activations
from .iq3s_int8t import _smem_base
from .iq3xxs import _popc, grid32
from .iq3xxs_int8 import BLOCK_BYTES
from .iq4xs_int8t import _lds_v


@g.jit
def iq3xxs_int8t_kernel(XQ, SX, WT, Y, GRID32,
                        M, N, num_k_blocks, stride_xq, stride_sx, stride_ym, stride_yk, P, C, stride_pm, stride_pk,
                        BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr, REDUCE: gl.constexpr, ROW_WORDS: gl.constexpr,
                        TW: gl.constexpr, NCH: gl.constexpr, SLOT: gl.constexpr, SMEM_OFF: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    smem_flat: gl.constexpr = SwizzledSharedLayout(4, 1, 1, [0])
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 4], [32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 8])
    Lrow_w: gl.constexpr = SliceLayout(1, Lw)
    Lrow_c: gl.constexpr = SliceLayout(0, mma)

    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    per = (num_k_blocks + SPLITK - 1) // SPLITK
    kb0 = pid_k * per
    kb1 = gl.minimum(kb0 + per, num_k_blocks)

    c16: gl.constexpr = BlockedLayout([4], [32], [4], [0])
    im = gl.arange(0, 2048, layout=c16)
    im = gl.max_contiguous(gl.multiple_of(im, 4), 4)
    WT32 = WT.to(gl.pointer_type(gl.int32), bitcast=True)
    tile0 = (pid_n * num_k_blocks).to(gl.int64) * TW
    smem = gl.allocate_shared_memory(gl.int32, [2, SLOT], smem_flat)

    # the field addresses: per row in the decode's row layout (aw) and in the accumulator's column layout (ac)
    rw = gl.arange(0, 64, layout=Lrow_w)
    aw = _smem_base(rw) + SMEM_OFF + rw * (ROW_WORDS * 4)
    rc = gl.arange(0, 64, layout=Lrow_c)
    ac = _smem_base(rc) + SMEM_OFF + rc * (ROW_WORDS * 4)
    bidx = gl.arange(0, 8, layout=SliceLayout(0, Lw))
    zero2 = gl.zeros([64, 8], gl.int32, Lw)
    b2 = zero2 + bidx[None, :]
    gsh2 = 7 * (b2 >> 1)                        # the group's 7 sign bits in the u32
    nsh2 = 4 * (b2 & 1)                         # the word's nibble in the group's 8-bit mask
    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    base0 = WT32 + tile0 + kb0 * TW
    for c in gl.static_range(NCH):
        if (c + 1) * 2048 <= TW:
            async_copy.async_copy_global_to_shared(smem.index(0).slice(c * 2048, 2048), base0 + c * 2048 + im)
        else:
            async_copy.async_copy_global_to_shared(smem.index(0).slice(c * 2048, 2048), base0 + c * 2048 + im, im < (TW - c * 2048))
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            base1 = WT32 + tile0 + (kb + 1) * TW
            for c in gl.static_range(NCH):
                if (c + 1) * 2048 <= TW:
                    async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(c * 2048, 2048), base1 + c * 2048 + im)
                else:
                    async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(c * 2048, 2048), base1 + c * 2048 + im, im < (TW - c * 2048))
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        so = (it % 2) * (SLOT * 4)
        dw = _lds_v(ac + so + 24 * 4)
        d = (dw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            qsw0 = _lds_v(aw + so + (2 * ib) * 4)                                # grid indices of words 0..3
            qsw1 = _lds_v(aw + so + (2 * ib + 1) * 4)                            # words 4..7
            auxw = _lds_v(aw + so + (16 + ib) * 4)                               # signs (7 bits per group) | scale << 28
            auxc = _lds_v(ac + so + (16 + ib) * 4)                               # the same word for the scale, column layout
            qsw0_2 = zero2 + qsw0[:, None]
            qsw1_2 = zero2 + qsw1[:, None]
            aux_2 = zero2 + auxw[:, None]
            idx = gl.where(b2 < 4, qsw0_2 >> (8 * b2), qsw1_2 >> (8 * (b2 - 4))) & 0xFF
            gw = gl.load(GRID32 + idx)
            s7 = (aux_2 >> gsh2) & 127
            mask8 = s7 | ((_popc(s7) & 1) << 7)
            nib = (mask8 >> nsh2) & 0xF
            mask4 = ((nib * 0x204081) & 0x01010101) * 0xFF
            wq = (gw ^ mask4) + (mask4 & 0x01010101)
            w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
            b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
            acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
            s = (auxc >> 28) & 0xF
            dl = d * (2.0 * s.to(gl.float32) + 1.0) * 0.25
            acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = pid_n * BN + gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < N)
    keep = smem.index(0).slice(0, 64).load(SliceLayout(0, mma))
    omask = omask & (keep[None, :] != 0x7FFFFFFF)
    if REDUCE and SPLITK > 1:
        # the fp32 partial to P[pid_k]; the last CTA of the tile sums the partials in order into Y (bf16)
        _splitk_reduce(Y, P, C, acc, ym, yn, omask, pid_n, pid_k, stride_ym, stride_pm, stride_pk, SPLITK)
    else:
        Yp = Y + pid_k.to(gl.int64) * stride_yk
        gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def repack_dlast_tiles(W: torch.Tensor, n_out: int, block: int, row_bytes: int) -> torch.Tensor:
    """Raw rows of a type whose block starts with a 2-byte d (uint8 [n_out, nb * block]) -> tile-major
    uint8 [n_tiles, nb, 64, row_bytes]: the block's bytes 2.. first, d in the two bytes after them, the
    rest zero; rows past n_out zero."""
    assert W.dtype == torch.uint8 and W.dim() == 2 and W.shape[0] == n_out and W.shape[1] % block == 0
    assert row_bytes >= block + 2 and row_bytes % 4 == 0
    nb = W.shape[1] // block
    n_tiles = -(-n_out // 64)
    t = W.reshape(n_out, nb, block)
    r = torch.zeros((n_tiles * 64, nb, row_bytes), dtype=torch.uint8, device=W.device)
    r[:n_out, :, :block - 2] = t[:, :, 2:]
    r[:n_out, :, block - 2:block] = t[:, :, :2]
    return r.reshape(n_tiles, 64, nb, row_bytes).permute(0, 2, 1, 3).contiguous()


def repack_iq3xxs_tiles(W: torch.Tensor, n_out: int, row_bytes: int = 100) -> torch.Tensor:
    return repack_dlast_tiles(W, n_out, BLOCK_BYTES, row_bytes)


def tile_geometry(row_bytes: int) -> tuple[int, int, int, int]:
    """(ROW_WORDS, TW, NCH, SLOT) of a 64-row tile: words per row, words per tile, 2048-word copy chunks, the
    power-of-two shared slot holding the chunks."""
    assert row_bytes % 4 == 0
    tw = 64 * row_bytes // 4
    nch = -(-tw // 2048)
    slot = max(1 << (tw - 1).bit_length(), nch * 2048)
    return row_bytes // 4, tw, nch, slot


def iq3xxs_int8t_gemm(WT: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                      quantized: tuple | None = None, smem_off: int = 0, reduce: bool = False, return_partials: bool = False, out: torch.Tensor | None = None):
    """WT the tile-major repack (repack_iq3xxs_tiles); quantized: (int8 X, scales[, sums])."""
    BN = 64
    assert WT.dtype == torch.uint8 and WT.is_cuda and WT.dim() == 4 and WT.is_contiguous() and WT.shape[2] == 64
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    nb = K // 256
    row_bytes = WT.shape[3]
    assert K % 256 == 0 and WT.shape[1] == nb and WT.shape[0] == -(-n_out // BN) and M <= 16 and row_bytes >= BLOCK_BYTES + 2
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations(X.contiguous())
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=row_bytes)   # 0034: the split capped by the batch rows
    splitk = max(1, min(splitk, nb))
    Y, P, C, strides = splitk_buffers(X, n_out, splitk, reduce, out)
    row_words, tw, nch, slot = tile_geometry(row_bytes)
    iq3xxs_int8t_kernel[(WT.shape[0], splitk)](
        XQ, SX, WT, Y, grid32(WT.device), M, n_out, nb, XQ.stride(0), SX.stride(0), strides[0], strides[1], P, C, strides[2], strides[3],
        BM=16, BN=BN, SPLITK=splitk, REDUCE=bool(reduce and splitk > 1), ROW_WORDS=row_words, TW=tw, NCH=nch, SLOT=slot, SMEM_OFF=smem_off, num_warps=num_warps)
    res = splitk_result(Y, X, splitk, reduce, out)
    return (res, P) if return_partials else res
