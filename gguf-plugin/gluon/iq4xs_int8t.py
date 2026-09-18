"""IQ4_XS x bf16 GEMM in Gluon, the int8 form on a tile-major repack with
16-byte copies, the pattern for the other types. The rows of 136 bytes (34
words, word-aligned already: d and
scales_h in word 0, scales_l in word 1, qs in words 2-33) are laid out
tile-major - a tile of 64 rows x ROW_BYTES per (64-row tile, k-block) - and
copied linearly in 16-byte chunks into a flat shared array (Gluon wants
power-of-two shapes, so the array is [2, 4096] words and the tile fills the
first TW); the fields are read as ld.shared gathers at row * ROW_WORDS + w,
which any row stride allows. ROW_BYTES 136 is the linear image (a column read
of 32 consecutive rows at stride 34 words hits 16 banks: 2-way conflicts);
140 (35 words, odd, +2.9 % bytes) is conflict-free. The decode (kvalues
through two prmt), the direct A loads, the mma and the epilogue are the int8
form's: bit-identical.

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

from .iq3s import _prmt, split_k_for
from .iq3s_int8 import _word_to_i8x4, quantize_activations
from .iq3s_int8t import _smem_base
from .iq4xs_int8 import BLOCK_BYTES, T0, T1, T2, T3, _kvalues_lookup


@g.jit
def _lds_v(addr):
    """ld.shared of one word per lane at a computed address; not pure, so it stays
    behind the barrier that orders it after the tile's copy."""
    return gl.inline_asm_elementwise("ld.shared.b32 $0, [$1];", "=r,r", [addr], dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def iq4xs_int8t_kernel(XQ, SX, WT, Y,
                       M, N, num_k_blocks, stride_xq, stride_sx, stride_ym, stride_yk, P, C, stride_pm, stride_pk,
                       T0: gl.constexpr, T1: gl.constexpr, T2: gl.constexpr, T3: gl.constexpr,
                       BM: gl.constexpr, BN: gl.constexpr, SPLITK: gl.constexpr, REDUCE: gl.constexpr, ROW_WORDS: gl.constexpr,
                       TW: gl.constexpr, NMAIN: gl.constexpr, RP: gl.constexpr, SLOT: gl.constexpr, SMEM_OFF: gl.constexpr):
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

    # the tile copy: NMAIN words in 16-byte chunks of a [2048]-word layout, the rest (RP words, masked to TW - NMAIN) after
    c16: gl.constexpr = BlockedLayout([4], [32], [4], [0])
    c1: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    im = gl.arange(0, 2048, layout=c16)
    im = gl.max_contiguous(gl.multiple_of(im, 4), 4)
    ir = gl.arange(0, RP, layout=c1)
    rmask = ir < (TW - NMAIN)
    WT32 = WT.to(gl.pointer_type(gl.int32), bitcast=True)
    tile0 = (pid_n * num_k_blocks).to(gl.int64) * TW
    smem = gl.allocate_shared_memory(gl.int32, [2, SLOT], smem_flat)

    # the field reads: ld.shared at (row * ROW_WORDS + w) words, per lane row in the two row layouts
    rw = gl.arange(0, 64, layout=Lrow_w)
    rc = gl.arange(0, 64, layout=Lrow_c)
    aw = _smem_base(rw) + SMEM_OFF + rw * (ROW_WORDS * 4)
    ac = _smem_base(rc) + SMEM_OFF + rc * (ROW_WORDS * 4)

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
    base0 = WT32 + tile0 + kb0 * TW
    for c in gl.static_range(NMAIN // 2048):
        async_copy.async_copy_global_to_shared(smem.index(0).slice(c * 2048, 2048), base0 + c * 2048 + im)
    if RP > 0:
        async_copy.async_copy_global_to_shared(smem.index(0).slice(NMAIN, RP), base0 + NMAIN + ir, rmask)
    async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if it + 1 < ntiles:
            base1 = WT32 + tile0 + (kb + 1) * TW
            for c in gl.static_range(NMAIN // 2048):
                async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(c * 2048, 2048), base1 + c * 2048 + im)
            if RP > 0:
                async_copy.async_copy_global_to_shared(smem.index((it + 1) % 2).slice(NMAIN, RP), base1 + NMAIN + ir, rmask)
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gl.barrier()
        so = (it % 2) * (SLOT * 4)
        # per row in the accumulator's column layout: d and scales_h (word 0), scales_l (word 1)
        w0 = _lds_v(ac + so)
        slw = _lds_v(ac + so + 4)
        d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
        shw = (w0 >> 16) & 0xFFFF
        for ib in gl.static_range(8):
            a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
            # the sub-block's four qs words per row (words 2 + 4 ib ..), the word b mod 4 per element
            q0 = _lds_v(aw + so + (2 + 4 * ib) * 4)
            q1 = _lds_v(aw + so + (3 + 4 * ib) * 4)
            q2 = _lds_v(aw + so + (4 + 4 * ib) * 4)
            q3 = _lds_v(aw + so + (5 + 4 * ib) * 4)
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
    # a Gluon-visible read of the array after the loop keeps it allocated (the asm reads are invisible to the liveness analysis)
    keep = smem.index(0).slice(0, 64).load(SliceLayout(0, mma))
    omask = omask & (keep[None, :] != 0x7FFFFFFF)
    if REDUCE and SPLITK > 1:
        # the fp32 partial to P[pid_k]; the last CTA of the tile sums the partials in order into Y (bf16)
        _splitk_reduce(Y, P, C, acc, ym, yn, omask, pid_n, pid_k, stride_ym, stride_pm, stride_pk, SPLITK)
    else:
        Yp = Y + pid_k.to(gl.int64) * stride_yk
        gl.store(Yp + ym[:, None] * stride_ym + yn[None, :], acc.to(Y.dtype.element_ty), mask=omask)


def repack_iq4xs_tiles(W: torch.Tensor, n_out: int, row_bytes: int = 140) -> torch.Tensor:
    """Raw IQ4_XS rows [n_out, nb * 136] (uint8) -> tile-major uint8 [n_tiles, nb, 64, row_bytes]
    (136: the rows as they are; 140: four bytes of padding for an odd word stride), rows past n_out zero."""
    assert W.dtype == torch.uint8 and W.dim() == 2 and W.shape[0] == n_out and W.shape[1] % BLOCK_BYTES == 0 and row_bytes >= BLOCK_BYTES and row_bytes % 4 == 0
    nb = W.shape[1] // BLOCK_BYTES
    n_tiles = -(-n_out // 64)
    t = W.reshape(n_out, nb, BLOCK_BYTES)
    if row_bytes > BLOCK_BYTES:
        t = torch.cat([t, torch.zeros((n_out, nb, row_bytes - BLOCK_BYTES), dtype=torch.uint8, device=W.device)], dim=-1)
    if n_tiles * 64 > n_out:
        t = torch.cat([t, torch.zeros((n_tiles * 64 - n_out, nb, row_bytes), dtype=torch.uint8, device=W.device)], dim=0)
    return t.reshape(n_tiles, 64, nb, row_bytes).permute(0, 2, 1, 3).contiguous()


def iq4xs_int8t_gemm(WT: torch.Tensor, X: torch.Tensor, n_out: int, splitk: int | None = None, num_warps: int = 4,
                     quantized: tuple[torch.Tensor, torch.Tensor] | None = None, smem_off: int = 0, reduce: bool = False, return_partials: bool = False, out: torch.Tensor | None = None):
    """WT the tile-major repack [n_tiles, nb, 64, row_bytes] (repack_iq4xs_tiles)."""
    BN = 64
    assert WT.dtype == torch.uint8 and WT.is_cuda and WT.dim() == 4 and WT.is_contiguous() and WT.shape[2] == 64
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    nb = K // 256
    row_bytes = WT.shape[3]
    assert K % 256 == 0 and WT.shape[1] == nb and WT.shape[0] == -(-n_out // BN) and M <= 16 and row_bytes % 4 == 0
    XQ, SX = quantized[:2] if quantized is not None else quantize_activations(X.contiguous())
    if splitk is None:
        splitk = split_k_for(nb, n_out, m=M, block_bytes=row_bytes)   # 0034: the split capped by the batch rows
    splitk = max(1, min(splitk, nb))
    Y, P, C, strides = splitk_buffers(X, n_out, splitk, reduce, out)
    tw = 64 * row_bytes // 4                       # words per tile
    slot = 1 << (tw - 1).bit_length()              # the flat array per stage, a power of two >= tw
    nmain = (tw // 2048) * 2048
    rem = tw - nmain
    rp = 0 if rem == 0 else 1 << (rem - 1).bit_length()
    iq4xs_int8t_kernel[(WT.shape[0], splitk)](
        XQ, SX, WT, Y, M, n_out, nb, XQ.stride(0), SX.stride(0), strides[0], strides[1], P, C, strides[2], strides[3],
        T0=T0, T1=T1, T2=T2, T3=T3, BM=16, BN=BN, SPLITK=splitk, REDUCE=bool(reduce and splitk > 1), ROW_WORDS=row_bytes // 4, TW=tw, NMAIN=nmain, RP=rp,
        SLOT=slot, SMEM_OFF=smem_off, num_warps=num_warps)
    res = splitk_result(Y, X, splitk, reduce, out)
    return (res, P) if return_partials else res
