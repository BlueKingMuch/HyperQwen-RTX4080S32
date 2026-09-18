"""One kernel, one launch for a layer of mixed-type shards: the shards'
tiles of all int8 types packed
into one buffer at load, a descriptor table with one row per CTA (the tile's
word offset, its type, row words, output column, valid rows, k-range,
split), and a kernel that walks the table - the type is uniform per CTA and
only selects the compiled decode. Every tile writes its 64 columns straight
into the layer's output (or its split's fp32 partial), split-K is planned
over the whole layer, the activations are quantised once per layer.

v2 (after the NCU cell of v1): every type's k-loop is its own
branch of a constexpr-typed helper, so the type and the row words are
compile-time inside the branch and only the descriptor's scalars live
across the branches (v1's single loop with the decode selected inside kept
every type's address vectors live together: 120-192 registers, the IQ2
branch +6-11 % instructions); the stage is 2,304 words (Q4_K's tile, the
largest) with the IQ3_S grid table in stage 0's free tail - 4,608 words,
18 KB, five blocks per SM at <= 102 registers (v1's [2, 4096] stage: three).
The arithmetic is a constexpr switch: E = 0 the per-32 arithmetic of every
tile kernel (bit-identical per shard to the separate launches at the same
split), E = 1 the per-256 sixth form on every type (the activation scale
per 256, the sub-block scales as integers on the int32 mma results, one
fp32 conversion per k-block; Q4_K's min term from the int32 sums of the
quantised activations per 32 - the same rounding the 0041 battery accepted
on IQ3_S).

    packed, meta = pack_layer([(W_q, n_q, 18), (W_k, n_k, 12), (W_v, n_v, 21)])   # raw rows, rows, type
    y = grouped_gemm(packed, meta, x, splitk=None, e=True)                          # bf16 [M, n_q + n_k + n_v]

This module adds the latency hiding, pack_tiles (the loader's tiles) and
prepare_grouped_layer (the loader's step).

The 32-row form: BM a launch-time choice of 16
or 32 (bm); at 32 every warp issues two mma row blocks per decoded B fragment, so a verify batch of 17-32 rows
(three or four streams under the 7-token drafter) streams the weights once instead of twice; the AMODE-2 A
stages doubled into a second allocation (regions 3 and 4); the keep-alive sentinel only in region 0; the split
tables to 32 rows, the budget's row count saturating at 16. At bm 16 the
launch is byte for byte the 16-row one.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, SliceLayout, DotOperandLayout, NVMMADistributedLayout, SwizzledSharedLayout,
    DistributedLinearLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from .iq2 import GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, _popc, grid_words
from .iq2_int8 import _b_operand, _negate_bytes
from .iq2_int8t import repack_iq2_tiles
from .iq3s import _prmt, grid32 as grid32_iq3s
from .iq3s_int8 import _word_to_i8x4, quantize_activations
from .iq3s_int8t import _lds, _smem_base, repack_iq3s_tiles
from .iq3xxs import grid32 as grid32_iq3xxs
from .iq3xxs_int8t import repack_iq3xxs_tiles
from .iq4xs_int8 import T0, T1, T2, T3, _kvalues_lookup
from .iq4xs_int8t import _lds_v, repack_iq4xs_tiles
from .q4k_int8t import repack_q4k_tiles
from .splitk import PARTIALS_BUDGET
from .tiles import TILE_TYPES

GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS = 12, 18, 21, 23
STAGE = 2304           # words per shared stage: the largest tile (Q4_K, 36 words x 64 rows)
TAB_W = 64 * 28        # the IQ3_S grid table's word offset: stage 0's free tail (the IQ3_S tile is 1,792 words; 512 free)
SMEM_WORDS = 2 * STAGE   # two allocations of 4096 + 512 words (the compiler's shapes are powers of two), one flat region:
#                          18,432 B + the 1 KB the driver reserves per block = 19.5 KB, five blocks in the SM's 100 KB
#                          (the table after the stages, 20.5 KB with the reservation, measured four)
# the shared regions (words): 0 the [4096] + [512] region with 2,304-word stages and the tables in stage 0's tail
# (the M = 1 variant, five blocks); 1 the [8192] region with the A stages at 6144 (AMODE 2 on a layer with an
# IQ4_XS or Q4_K shard, or three stages; three blocks); 2 the compact [4096] + [2048] region for layers whose
# tiles are at most 1,792 words (IQ3_S, IQ3_XXS, the IQ2 types): stages at 0 / 1792, the tables at 3584, the A
# stages in the second allocation - 24 KB, four blocks at M = 8
REGIONS = {0: dict(stage=2304, tab=1792, tabx=1600, aoff=0), 1: dict(stage=2304, tab=1792, tabx=1600, aoff=6144), 2: dict(stage=1792, tab=3584, tabx=3584, aoff=4096),
           # 0046: the 32-row A stages (2 x 32 x 64 words) in the second allocation - 3 the [8192] + [4096] region (48 KB, two
           # blocks) for layers with an IQ4_XS or Q4_K shard, 4 the compact [4096] + [4096] region (32 KB, three blocks)
           3: dict(stage=2304, tab=1792, tabx=1600, aoff=0), 4: dict(stage=1792, tab=3584, tabx=3584, aoff=0)}
DESC_W = 16            # int32 words per descriptor row
# the descriptor row
D_OFF, D_TYPE, D_RW, D_COL0, D_NVALID, D_KB0, D_KB1, D_SPLIT, D_TW = range(9)
ROW_WORDS = {wt: rb // 4 for wt, (_, rb, _) in TILE_TYPES.items()}
assert all(64 * rw <= STAGE and (64 * rw <= 2048 or (64 * rw - 2048) % 128 == 0) for rw in ROW_WORDS.values())   # IQ4_XS 2176, Q4_K 2304


# ---- the activations per 256 (the sixth form's convention), with the int32 sums per 32 for the K types' mins

@triton.jit
def _quantize256_kernel(X, XQ, SX, SUMX, K, stride_x, stride_xq, stride_sx, stride_sum, WITH_SUMS: tl.constexpr):
    """One program per (row, block of 256 columns): the block to int8 with its absmax / 127, the scale per
    block, and (WITH_SUMS) the eight int32 sums of the quantised values per 32 (the same rounding as
    iq3s_int8e_gluon's quantiser: bit-identical int8 and scales)."""
    m = tl.program_id(0)
    b = tl.program_id(1)
    offs = b * 256 + tl.arange(0, 256)
    x = tl.load(X + m * stride_x + offs, mask=offs < K, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    sx = tl.where(amax > 0, amax / 127.0, 1.0)
    q = x / sx
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)
    tl.store(XQ + m * stride_xq + offs, q.to(tl.int8), mask=offs < K)
    tl.store(SX + m * stride_sx + b, sx)
    if WITH_SUMS:
        s = tl.sum(tl.reshape(q, [8, 32]), axis=1)
        tl.store(SUMX + m * stride_sum + b * 8 + tl.arange(0, 8), s)


def quantize_activations_256(X: torch.Tensor, with_sums: bool = False):
    """X bf16 [M, K] -> (int8 [M, K], fp32 scales [M, K // 256][, int32 sums per 32 [M, K // 32]])."""
    M, K = X.shape
    assert K % 256 == 0
    XQ = torch.empty((M, K), dtype=torch.int8, device=X.device)
    SX = torch.empty((M, K // 256), dtype=torch.float32, device=X.device)
    SUMX = torch.empty((M, K // 32), dtype=torch.int32, device=X.device) if with_sums else SX
    _quantize256_kernel[(M, K // 256)](X, XQ, SX, SUMX, K, X.stride(0), XQ.stride(0), SX.stride(0), SUMX.stride(0),
                                       WITH_SUMS=with_sums, num_warps=2)
    return (XQ, SX, SUMX) if with_sums else (XQ, SX)


@triton.jit
def _sum_partials_kernel(P, Y, S, n, stride_pk, BLOCK: tl.constexpr):
    """Y[i] = sum_s P[s, i] in fp32, rounded once to Y's type (torch.sum(P, 0, dtype=bf16) casts every partial
    to bf16 before adding - up to S + 1 half-ulps of error on the split-K path)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(S):
        acc += tl.load(P + s * stride_pk + offs, mask=m, other=0.0)
    tl.store(Y + offs, acc.to(Y.dtype.element_ty), mask=m)


def sum_partials(P: torch.Tensor, out: torch.Tensor | None = None, dtype=torch.bfloat16) -> torch.Tensor:
    """P fp32 [S, M, N] contiguous -> [M, N] in dtype (into out, contiguous, if given)."""
    S, M, N = P.shape
    assert P.is_contiguous()
    Y = out if out is not None else torch.empty((M, N), dtype=dtype, device=P.device)
    assert Y.is_contiguous() and Y.shape == (M, N)
    n = M * N
    _sum_partials_kernel[(triton.cdiv(n, 1024),)](P, Y, S, n, P.stride(0), BLOCK=1024, num_warps=4)
    return Y


# ---- the packing (at load) and the descriptors (per split)

def _repack(W: torch.Tensor, n_out: int, wt: int) -> torch.Tensor:
    if wt == GGML_TYPE_IQ3_S:
        return repack_iq3s_tiles(W, n_out)
    if wt == GGML_TYPE_IQ4_XS:
        return repack_iq4xs_tiles(W, n_out, 136)
    if wt == GGML_TYPE_Q4_K:
        return repack_q4k_tiles(W, n_out, 144)
    if wt == GGML_TYPE_IQ3_XXS:
        return repack_iq3xxs_tiles(W, n_out, 100)
    return repack_iq2_tiles(W, n_out, wt)


def pack_tiles(shards: list[tuple[torch.Tensor, int, int]]):
    """[(tiles uint8 [n_tiles, nb, 64, row_bytes] - the type's repack, n_out, type)] -> (packed uint8 buffer,
    meta): every shard's tiles contiguous in the buffer; meta per shard: (base_words, type, row_words, n_out,
    n_tiles, col0); max_rw the layer's largest row words (the region's choice)."""
    parts, meta, base_words, col0 = [], [], 0, 0
    nb = None
    for t, n_out, wt in shards:
        block, row_bytes, _ = TILE_TYPES[wt]
        assert t.dtype == torch.uint8 and t.dim() == 4 and t.shape[2] == 64 and t.shape[3] == row_bytes and t.is_contiguous()
        assert t.shape[0] == -(-n_out // 64) and (nb is None or t.shape[1] == nb)
        nb = int(t.shape[1])
        parts.append(t.reshape(-1))
        meta.append((base_words, wt, row_bytes // 4, n_out, int(t.shape[0]), col0))
        base_words += t.numel() // 4
        col0 += n_out
    packed = torch.cat(parts)
    return packed, {"shards": meta, "nb": nb, "n_total": col0, "words": base_words, "max_rw": max(m[2] for m in meta)}


def pack_layer(shards: list[tuple[torch.Tensor, int, int]]):
    """[(raw rows uint8 [n_out, nb * block], n_out, type)] -> (packed uint8 buffer, meta), the repack included."""
    return pack_tiles([(_repack(W, n_out, wt), n_out, wt) for W, n_out, wt in shards])


# the split-K of a grouped launch per (output tiles, k-blocks): from a sweep (E = 1, base clock,
# caches flushed, M = 1 and 8), the split that is best or within 2 % of
# the best at both M; else about 4.5 waves of CTAs at four blocks per SM (the M = 8 variant's occupancy), never
# more than half the k-blocks or 8
GROUPED_SPLITS = {
    (80, 68): 5,     # [5120x17408] ffn_down: 65.3 / 74.6 us at M = 1 / 8 (the best 65.3 / 73.7)
    (96, 20): 8,     # [6144x5120] attn_gate: 29.7 / 31.8 (the best 29.2 / 31.8)
    (80, 24): 3,     # [5120x6144] ssm_out: 29.2 / 32.7
    (160, 20): 8,    # [10240x5120] attn_qkv: 43.9 / 46.5 (the best 43.8 / 46.5)
    (192, 20): 5,    # [12288x5120] attn_q alone (the q/k/v cell's split)
    (272, 20): 8,    # [17408x5120] ffn_gate / up alone: 63.7 / 67.9
    (224, 20): 5,    # [14336x5120] q/k/v grouped: 57.6 / 61.6 (the best 56.1 / 60.5)
    (32, 20): 5,     # [2048x5120] k/v grouped: 15.5 / 17.2
    (16, 20): 5,     # [1024x5120] k or v alone
    (544, 20): 5,    # [34816x5120] gate/up grouped: 132.3 / 131.4 (the IQ2 pair 92.8 / 112.9)
    (3880, 20): 1,   # [248320x5120] the head: 16 waves as it is
}


GROUPED_SPLITS_M32 = {   # 0046: the split at 17-32 rows per (n_tiles, nb), swept in the step's state (lab m32_split_column.py)
    (32, 20): 5,
    (80, 24): 2,
    (80, 68): 5,
    (96, 20): 5,
    (160, 20): 1,
    (224, 20): 5,
    (272, 20): 2,
    (544, 20): 1,
    (3880, 20): 1,
}


def grouped_split_for(n_tiles: int, nb: int, m: int | None = None, block_bytes: int | None = None) -> int:
    """The split-K of a grouped launch of n_tiles output tiles over nb k-blocks: the measured table, else the
    formula; with m (the batch rows) and block_bytes (the layer's bytes per 256 weights) capped so that the
    fp32 partial sums stay within the recipe's budget (splitk.py)."""
    s = GROUPED_SPLITS.get((n_tiles, nb))
    if m is not None and m > 16 and (n_tiles, nb) in GROUPED_SPLITS_M32:
        return max(1, min(GROUPED_SPLITS_M32[(n_tiles, nb)], nb))   # 0046: the swept 32-row column, no cap
    if s is None:
        s = int(max(1, min(8, nb // 2, round(1440 / n_tiles))))
    if m is not None and block_bytes is not None and m > 0:
        bits = 8 * block_bytes / 256
        cap = int(PARTIALS_BUDGET * (nb * 256) * bits / (64 * min(m, 16)))   # 0046: the budget's row count saturates at 16
        s = min(s, max(1, cap))
    return max(1, min(s, nb))


def prepare_grouped_layer(tiles_list: list[torch.Tensor], n_outs: list[int], wtypes: list[int], ms=range(1, 33)):
    """The loader's step for a run of tile-type shards: (packed buffer, the shards' tiles as views into it, the
    descriptor tables of the splits the launch can choose for batches of 1..16 rows, those splits, nb, the
    layer's bytes per 256 weights). The views replace the separate tile tensors (the prefill path reads them)."""
    packed, meta = pack_tiles(list(zip(tiles_list, n_outs, wtypes)))
    views = []
    for (base_words, wt, rw, n_out, n_tiles, col0), t in zip(meta["shards"], tiles_list):
        views.append(packed[base_words * 4: base_words * 4 + t.numel()].view(n_tiles, meta["nb"], 64, rw * 4))
    nb, n_total = meta["nb"], meta["n_total"]
    n_tiles = sum(-(-n // 64) for n in n_outs)
    block_bytes = int(round(packed.numel() / (n_total * nb)))
    splits = sorted({grouped_split_for(n_tiles, nb, m=m, block_bytes=block_bytes) for m in ms})
    descs = [descriptors(meta, s, packed.device) for s in splits]
    return packed, views, descs, splits, nb, block_bytes


def descriptors(meta, splitk: int, device) -> torch.Tensor:
    # cached inside the layer's own meta (a module-level cache keyed by id(meta) served a freed layer's table
    # to the next layer whose dict got the same id: the NaN cells of the first test run)
    cache = meta.setdefault("desc", {})
    if splitk in cache:
        return cache[splitk]
    nb = meta["nb"]
    assert 1 <= splitk <= nb, (splitk, nb)
    rows = []
    for base_words, wt, rw, n_out, n_tiles, col0 in meta["shards"]:
        tw = 64 * rw
        for i in range(n_tiles):
            for s in range(splitk):
                # 0044: the balanced partition - every split index owns a non-empty range. The ceil partition of
                # 0042 (per = ceil(nb / splitk), the indices with kb0 >= nb skipped) left the last index without a
                # CTA when ceil(nb / splitk) * (splitk - 1) >= nb (20 k-blocks over 8 or 6 splits: the 6144- and
                # 8192-wide projections at their table splits), so that partial slice was never written and the
                # fp32 sum added torch.empty's contents to the output (found by a v2p identity run).
                kb0 = (s * nb) // splitk
                kb1 = ((s + 1) * nb) // splitk
                assert kb1 > kb0, (s, splitk, nb)
                r = [0] * DESC_W
                r[D_OFF] = base_words + i * nb * tw
                r[D_TYPE], r[D_RW], r[D_COL0] = wt, rw, col0 + i * 64
                r[D_NVALID] = min(64, n_out - i * 64)
                r[D_KB0], r[D_KB1], r[D_SPLIT], r[D_TW] = kb0, kb1, s, tw
                rows.append(r)
    d = torch.tensor(rows, dtype=torch.int32, device=device)
    cache[splitk] = d
    return d


# ---- the kernel

@g.jit
def _cp_async16(saddr, gaddr, pred):
    """One predicated 16-byte cp.async per element: saddr the shared byte address, gaddr the global address,
    pred 0 / 1 (a chunk past the tile is not issued - no zero fill, nothing written past the tile's words)."""
    return gl.inline_asm_elementwise(
        "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\n@p cp.async.cg.shared.global [$1], [$2], 16;\nmov.u32 $0, 0;\n}",
        "=r,r,l,r", [saddr, gaddr, pred], dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _sts(addr, val):
    return gl.inline_asm_elementwise("st.shared.b32 [$1], $2;\nmov.u32 $0, 0;", "=r,r,r", [addr, val], dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _copy_stage(sb, s, src, ic, TW: gl.constexpr, STAGE_: gl.constexpr):
    """The tile's TW words from src (an int32 pointer) into stage s of the flat shared region (sb its base
    address per element) in 16-byte chunks: ic the chunk ids (512 or 1024 per stage, one 16-byte chunk each),
    the chunks past TW not issued. The stage is any size (2,304 words here): the copy goes through raw shared
    addresses, as the gathers do, so the compiler's power-of-two shapes do not bind the stage."""
    gaddr = (src + ic * 4).to(gl.int64, bitcast=True)
    saddr = sb + (s * STAGE_ + ic * 4) * 4
    pred = (ic * 4 < TW).to(gl.int32)
    _cp_async16(saddr, gaddr, pred)


@g.jit
def _decode_half_iq3s(qsw, qh, sgw, selq, shq, shs, zero4, tbase):
    qs_4 = zero4 + qsw[:, None]
    qh_4 = zero4 + qh[:, None]
    sg_4 = zero4 + sgw[:, None]
    qsb = _prmt(qs_4, zero4, selq)
    idx = qsb | ((qh_4 << shq) & 0x100)
    gw = _lds(tbase + idx * 4)
    nib = (sg_4 >> shs) & 0xF
    x = (nib * 0x204081) & 0x01010101
    return (gw ^ (x * 0xFF)) + x


@g.jit
def _run(TP: gl.constexpr, RW: gl.constexpr, E: gl.constexpr, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split,
         XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
         M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk,
         BM: gl.constexpr, BN: gl.constexpr, STAGE_: gl.constexpr, TAB: gl.constexpr,
         K0: gl.constexpr, K1: gl.constexpr, K2: gl.constexpr, K3: gl.constexpr,
         AMODE: gl.constexpr, STAGES: gl.constexpr, GX: gl.constexpr, REGION: gl.constexpr, TABX: gl.constexpr, AOFF: gl.constexpr):
    """One CTA's tile of type TP (RW words per row): the k-loop over kb0..kb1 on the two-stage pipeline, the
    decode of TP, the epilogue store into the layer's output columns col0.. (or the split's partial)."""
    TW: gl.constexpr = 64 * RW
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    da8: gl.constexpr = DotOperandLayout(0, mma, 4)
    db8: gl.constexpr = DotOperandLayout(1, mma, 4)
    Lw: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 4], [32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 8])
    Lrow_w: gl.constexpr = SliceLayout(1, Lw)
    L4: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[32, 0]],
        lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 4])
    Lrow4: gl.constexpr = SliceLayout(1, L4)
    Lrow_c: gl.constexpr = SliceLayout(0, mma)

    NC: gl.constexpr = 1024 if TW > 2048 else 512
    cc: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    ic = gl.arange(0, NC, layout=cc)
    sb = _smem_base(ic)
    base = TILES32 + off.to(gl.int64)
    if TP == 21:
        tl1: gl.constexpr = BlockedLayout([4], [32], [4], [0])
        ti = gl.arange(0, 512, layout=tl1)
        _sts(_smem_base(ti) + (TAB + ti) * 4, gl.load(GRID_S + ti))
    if TP == 18 and GX == 1:
        tlx: gl.constexpr = BlockedLayout([2], [32], [4], [0])
        tix = gl.arange(0, 256, layout=tlx)
        _sts(_smem_base(tix) + (TABX + tix) * 4, gl.load(GRID_X + tix))

    # the field addresses: per row in the decode's row layout (aw), the accumulator's column layout (ac),
    # the IQ3_S [64, 4] row layout (a4), the Q4_K [64, 8] group gather (a2)
    rwv = gl.arange(0, 64, layout=Lrow_w)
    rcv = gl.arange(0, 64, layout=Lrow_c)
    r4v = gl.arange(0, 64, layout=Lrow4)
    aw = _smem_base(rwv) + rwv * (RW * 4)
    ac = _smem_base(rcv) + rcv * (RW * 4)
    a4 = _smem_base(r4v) + r4v * (RW * 4)
    r2 = gl.arange(0, 64, layout=SliceLayout(1, Lw))
    c2 = gl.arange(0, 8, layout=SliceLayout(0, Lw))
    a2 = _smem_base(r2)[:, None] + (r2[:, None] * RW + 4 + c2[None, :]) * 4
    bidx = gl.arange(0, 8, layout=SliceLayout(0, Lw))
    zero2 = gl.zeros([64, 8], gl.int32, Lw)
    b2 = zero2 + bidx[None, :]
    zero4 = gl.zeros([64, 4], gl.int32, L4)
    b4 = zero4 + gl.arange(0, 4, layout=SliceLayout(0, L4))[None, :]
    tbase = _smem_base(zero4) + TAB * 4
    mm = gl.arange(0, BM, layout=SliceLayout(1, da8))
    kx = gl.arange(0, 32, layout=SliceLayout(0, da8))
    m_ok = mm < M
    xrow = XQ + mm[:, None] * stride_xq
    ms = gl.arange(0, BM, layout=SliceLayout(1, mma))
    ms_ok = ms < M
    sxrow = SX + ms * stride_sx
    sumrow = SUMX + ms * stride_sum

    tbx = _smem_base(zero2) + TABX * 4
    if AMODE == 2:
        # the activation tile of a k-block: BM rows x 64 words, 4-byte cp.async coalesced along the row (16 lanes
        # x 4 B per row, two rows per warp), rows >= M masked; two stages after the weight stages in the region
        smem_a_words: gl.constexpr = SwizzledSharedLayout(1, 1, 8, [1, 0])
        smem_a8: gl.constexpr = SwizzledSharedLayout(4, 1, 8, [1, 0])
        cpa: gl.constexpr = BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0])
        am = gl.arange(0, BM, layout=SliceLayout(1, cpa))
        aw_ = gl.arange(0, 64, layout=SliceLayout(0, cpa))
        XQ32 = XQ.to(gl.pointer_type(gl.int32), bitcast=True)
        arow = am.to(gl.int64) * (stride_xq // 4)
        amask = (am < M)[:, None] & (aw_ < 64)[None, :]
        if REGION == 1:
            smem_a = smem.slice(AOFF, 2048)._reinterpret(gl.int32, [2, BM, 64], smem_a_words)
        else:
            smem_a = smem2._reinterpret(gl.int32, [2, BM, 64], smem_a_words)

    acc = gl.zeros([BM, BN], gl.float32, mma)
    ntiles = kb1 - kb0
    _copy_stage(sb, 0, base + kb0 * TW, ic, TW, STAGE_)
    if AMODE == 2:
        async_copy.async_copy_global_to_shared(smem_a.index(0), XQ32 + arow[:, None] + (kb0 * 64 + aw_)[None, :], amask)
    async_copy.commit_group()
    if STAGES == 3:
        if ntiles > 1:
            _copy_stage(sb, 1, base + (kb0 + 1) * TW, ic, TW, STAGE_)
        async_copy.commit_group()
    for it in range(0, ntiles):
        kb = kb0 + it
        if STAGES == 3:
            if it + 2 < ntiles:
                _copy_stage(sb, (it + 2) % 3, base + (kb + 2) * TW, ic, TW, STAGE_)
                async_copy.commit_group()
                async_copy.wait_group(2)
            elif it + 1 < ntiles:
                async_copy.wait_group(1)
            else:
                async_copy.wait_group(0)
        else:
            if it + 1 < ntiles:
                _copy_stage(sb, (it + 1) % 2, base + (kb + 1) * TW, ic, TW, STAGE_)
                if AMODE == 2:
                    async_copy.async_copy_global_to_shared(smem_a.index((it + 1) % 2), XQ32 + arow[:, None] + ((kb + 1) * 64 + aw_)[None, :], amask)
                async_copy.commit_group()
                async_copy.wait_group(1)
            else:
                async_copy.wait_group(0)
        gl.barrier()
        so = (it % STAGES) * (STAGE_ * 4)
        if AMODE == 1:
            # the eight fragments of this k-block, loaded together ahead of the decode (one latency per k-block)
            af0 = gl.load(xrow + (kb * 256 + 0 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            af1 = gl.load(xrow + (kb * 256 + 1 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            af2 = gl.load(xrow + (kb * 256 + 2 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            af3 = gl.load(xrow + (kb * 256 + 3 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            af4 = gl.load(xrow + (kb * 256 + 4 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            af5 = gl.load(xrow + (kb * 256 + 5 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            af6 = gl.load(xrow + (kb * 256 + 6 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
            af7 = gl.load(xrow + (kb * 256 + 7 * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
        if AMODE == 2:
            a_tile = smem_a.index(it % 2)._reinterpret(gl.int8, [BM, 256], smem_a8)
        if E == 1:
            sx = gl.load(sxrow + kb, mask=ms_ok, other=0.0)                       # the activation scale per 256
            acc_k = gl.zeros([BM, BN], gl.int32, mma)
        if TP == 21:
            # IQ3_S: qs words 0-15, qh 16-17, signs 18-25, scales 26, d 27 (the 0036 tile row)
            qh01 = _lds_v(a4 + so + 16 * 4)
            qh23 = _lds_v(a4 + so + 17 * 4)
            dw = _lds_v(ac + so + 27 * 4)
            d = (dw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
            scw = _lds_v(ac + so + 26 * 4)
            for ib in gl.static_range(8):
                if AMODE == 0:
                    a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
                elif AMODE == 1:
                    if ib == 0:
                        a8 = af0
                    elif ib == 1:
                        a8 = af1
                    elif ib == 2:
                        a8 = af2
                    elif ib == 3:
                        a8 = af3
                    elif ib == 4:
                        a8 = af4
                    elif ib == 5:
                        a8 = af5
                    elif ib == 6:
                        a8 = af6
                    else:
                        a8 = af7
                else:
                    a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)
                qsw0 = _lds_v(a4 + so + (2 * ib) * 4)
                qsw1 = _lds_v(a4 + so + (2 * ib + 1) * 4)
                sgw = _lds_v(a4 + so + (18 + ib) * 4)
                if ib < 4:
                    qh = (qh01 >> (8 * ib)) & 0xFF
                else:
                    qh = (qh23 >> (8 * (ib - 4))) & 0xFF
                wq_lo = _decode_half_iq3s(qsw0, qh, sgw, b4 | 0x4440, 8 - b4, 4 * b4, zero4, tbase)
                wq_hi = _decode_half_iq3s(qsw1, qh, sgw, b4 | 0x4440, 4 - b4, 16 + 4 * b4, zero4, tbase)
                w8 = gl.join(wq_lo, wq_hi)
                w4 = _word_to_i8x4(gl.join(gl.join(w8, w8), gl.join(w8, w8)))
                b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (2, 1, 3, 4, 0)), [32, 64]), db8)
                acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
                s = (scw >> (4 * ib)) & 0xF
                if E == 1:
                    si = 1 + 2 * s
                    acc_k = acc_k + acc_i * si[None, :]
                else:
                    sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                    dl = d * (1.0 + 2.0 * s.to(gl.float32))
                    acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
            if E == 1:
                acc = acc + acc_k.to(gl.float32) * (sx[:, None] * d[None, :])
        elif TP == 23:
            # IQ4_XS: d | scales_h (word 0), scales_l (word 1), qs words 2-33
            w0 = _lds_v(ac + so)
            slw = _lds_v(ac + so + 4)
            d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
            shw = (w0 >> 16) & 0xFFFF
            nsh2 = (b2 >> 2) * 4
            for ib in gl.static_range(8):
                if AMODE == 0:
                    a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
                elif AMODE == 1:
                    if ib == 0:
                        a8 = af0
                    elif ib == 1:
                        a8 = af1
                    elif ib == 2:
                        a8 = af2
                    elif ib == 3:
                        a8 = af3
                    elif ib == 4:
                        a8 = af4
                    elif ib == 5:
                        a8 = af5
                    elif ib == 6:
                        a8 = af6
                    else:
                        a8 = af7
                else:
                    a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)
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
                nib4 = (qw >> nsh2) & 0x0F0F0F0F
                wq = _kvalues_lookup(nib4, K0, K1, K2, K3)
                w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
                b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
                acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
                ls = ((slw >> (4 * ib)) & 0xF) | (((shw >> (2 * ib)) & 3) << 4)
                if E == 1:
                    si = ls - 32
                    acc_k = acc_k + acc_i * si[None, :]
                else:
                    sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                    dl = d * (ls.to(gl.float32) - 32.0)
                    acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
            if E == 1:
                acc = acc + acc_k.to(gl.float32) * (sx[:, None] * d[None, :])
        elif TP == 12:
            # Q4_K: d | dmin (word 0), scale / min bytes (words 1-3), qs words 4-35
            w0 = _lds_v(ac + so)
            sw1 = _lds_v(ac + so + 4)
            sw2 = _lds_v(ac + so + 8)
            sw3 = _lds_v(ac + so + 12)
            d = (w0 & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
            dmin = ((w0 >> 16) & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
            if E == 1:
                accm_k = gl.zeros([BM, BN], gl.int32, mma)
            for ib in gl.static_range(8):
                if ib % 2 == 0:
                    qw8 = _lds_v(a2 + so + (ib // 2) * 32)
                if AMODE == 0:
                    a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
                elif AMODE == 1:
                    if ib == 0:
                        a8 = af0
                    elif ib == 1:
                        a8 = af1
                    elif ib == 2:
                        a8 = af2
                    elif ib == 3:
                        a8 = af3
                    elif ib == 4:
                        a8 = af4
                    elif ib == 5:
                        a8 = af5
                    elif ib == 6:
                        a8 = af6
                    else:
                        a8 = af7
                else:
                    a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)
                if ib < 4:
                    sc = (sw1 >> (8 * ib)) & 63
                    mn = (sw2 >> (8 * ib)) & 63
                else:
                    sc = ((sw3 >> (8 * (ib - 4))) & 0xF) | (((sw1 >> (8 * (ib - 4) + 6)) & 3) << 4)
                    mn = ((sw3 >> (8 * (ib - 4) + 4)) & 0xF) | (((sw2 >> (8 * (ib - 4) + 6)) & 3) << 4)
                wq = (qw8 >> (4 * (ib % 2))) & 0x0F0F0F0F
                w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
                b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
                acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
                if E == 1:
                    sumx = gl.load(sumrow + kb * 8 + ib, mask=ms_ok, other=0)     # int32 sum of the 32 quantised activations
                    acc_k = acc_k + acc_i * sc[None, :]
                    accm_k = accm_k + sumx[:, None] * mn[None, :]
                else:
                    sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                    sumx = gl.load(sumrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                    dl = d * sc.to(gl.float32)
                    ml = dmin * mn.to(gl.float32)
                    acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :]) - sumx[:, None] * ml[None, :]
            if E == 1:
                acc = acc + acc_k.to(gl.float32) * (sx[:, None] * d[None, :]) - accm_k.to(gl.float32) * (sx[:, None] * dmin[None, :])
        elif TP == 18:
            # IQ3_XXS: qs words 0-15, aux 16-23, d word 24
            gsh2 = 7 * (b2 >> 1)
            nsh2x = 4 * (b2 & 1)
            dw = _lds_v(ac + so + 24 * 4)
            d = (dw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
            for ib in gl.static_range(8):
                if AMODE == 0:
                    a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
                elif AMODE == 1:
                    if ib == 0:
                        a8 = af0
                    elif ib == 1:
                        a8 = af1
                    elif ib == 2:
                        a8 = af2
                    elif ib == 3:
                        a8 = af3
                    elif ib == 4:
                        a8 = af4
                    elif ib == 5:
                        a8 = af5
                    elif ib == 6:
                        a8 = af6
                    else:
                        a8 = af7
                else:
                    a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)
                qsw0 = _lds_v(aw + so + (2 * ib) * 4)
                qsw1 = _lds_v(aw + so + (2 * ib + 1) * 4)
                auxw = _lds_v(aw + so + (16 + ib) * 4)
                auxc = _lds_v(ac + so + (16 + ib) * 4)
                qsw0_2 = zero2 + qsw0[:, None]
                qsw1_2 = zero2 + qsw1[:, None]
                aux_2 = zero2 + auxw[:, None]
                idx = gl.where(b2 < 4, qsw0_2 >> (8 * b2), qsw1_2 >> (8 * (b2 - 4))) & 0xFF
                if GX == 1:
                    gw = _lds(tbx + idx * 4)
                else:
                    gw = gl.load(GRID_X + idx)
                s7 = (aux_2 >> gsh2) & 127
                mask8 = s7 | ((_popc(s7) & 1) << 7)
                nib = (mask8 >> nsh2x) & 0xF
                mask4 = ((nib * 0x204081) & 0x01010101) * 0xFF
                wq = (gw ^ mask4) + (mask4 & 0x01010101)
                w4 = _word_to_i8x4(gl.join(gl.join(wq, wq), gl.join(wq, wq)))
                b8 = gl.convert_layout(gl.reshape(gl.permute(w4, (1, 2, 3, 0)), [32, 64]), db8)
                acc_i = mma_v2(a8, b8, gl.zeros([BM, BN], gl.int32, mma))
                s = (auxc >> 28) & 0xF
                if E == 1:
                    si = 2 * s + 1
                    acc_k = acc_k + acc_i * si[None, :]
                else:
                    sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                    dl = d * (2.0 * s.to(gl.float32) + 1.0) * 0.25
                    acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
            if E == 1:
                acc = acc + acc_k.to(gl.float32) * (sx[:, None] * (d * 0.25)[None, :])
        elif TP == 16:
            # IQ2_XXS: qs / aux word pairs 0-15, d at word 16
            g2 = b2 >> 1
            w2 = b2 & 1
            nsh2i = 4 * w2
            dw = _lds_v(ac + so + 16 * 4)
            d = (dw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
            for ib in gl.static_range(8):
                if AMODE == 0:
                    a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
                elif AMODE == 1:
                    if ib == 0:
                        a8 = af0
                    elif ib == 1:
                        a8 = af1
                    elif ib == 2:
                        a8 = af2
                    elif ib == 3:
                        a8 = af3
                    elif ib == 4:
                        a8 = af4
                    elif ib == 5:
                        a8 = af5
                    elif ib == 6:
                        a8 = af6
                    else:
                        a8 = af7
                else:
                    a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)
                f0 = _lds_v(aw + so + (2 * ib) * 4)
                aux = _lds_v(aw + so + (2 * ib + 1) * 4)
                auxc = _lds_v(ac + so + (2 * ib + 1) * 4)
                f0_2 = zero2 + f0[:, None]
                aux_2 = zero2 + aux[:, None]
                idx = (f0_2 >> (8 * g2)) & 0xFF
                s7 = (aux_2 >> (7 * g2)) & 127
                mask8 = s7 | ((_popc(s7) & 1) << 7)
                gw = gl.load(GRID2_XXS + idx * 2 + w2)
                wq = _negate_bytes(gw, (mask8 >> nsh2i) & 0xF)
                acc_i = mma_v2(a8, _b_operand(wq, db8), gl.zeros([BM, BN], gl.int32, mma))
                s = (auxc >> 28) & 0xF
                if E == 1:
                    si = 2 * s + 1
                    acc_k = acc_k + acc_i * si[None, :]
                else:
                    sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                    dl = d * (2.0 * s.to(gl.float32) + 1.0) * 0.125
                    acc = acc + acc_i.to(gl.float32) * (sx[:, None] * dl[None, :])
            if E == 1:
                acc = acc + acc_k.to(gl.float32) * (sx[:, None] * (d * 0.125)[None, :])
        else:
            # IQ2_XS (17): qs words 0-15, scales 16-17, d 18; IQ2_S (22): qs 0-7, signs 8-15, qh 16-17, scales 18-19, d 20
            g2 = b2 >> 1
            w2 = b2 & 1
            nsh2i = 4 * w2
            if TP == 17:
                dw = _lds_v(ac + so + 18 * 4)
            else:
                dw = _lds_v(ac + so + 20 * 4)
            d = (dw & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
            for ib in gl.static_range(8):
                if AMODE == 0:
                    a8 = gl.load(xrow + (kb * 256 + ib * 32 + kx)[None, :], mask=m_ok[:, None], other=0)
                elif AMODE == 1:
                    if ib == 0:
                        a8 = af0
                    elif ib == 1:
                        a8 = af1
                    elif ib == 2:
                        a8 = af2
                    elif ib == 3:
                        a8 = af3
                    elif ib == 4:
                        a8 = af4
                    elif ib == 5:
                        a8 = af5
                    elif ib == 6:
                        a8 = af6
                    else:
                        a8 = af7
                else:
                    a8 = a_tile.slice(ib * 32, 32, dim=1).load(da8)
                if TP == 17:
                    f0 = _lds_v(aw + so + (2 * ib) * 4)
                    f1 = _lds_v(aw + so + (2 * ib + 1) * 4)
                    scb = (_lds_v(ac + so + (16 + ib // 4) * 4) >> (8 * (ib % 4))) & 0xFF
                    f0_2 = zero2 + f0[:, None]
                    f1_2 = zero2 + f1[:, None]
                    q2 = gl.where(g2 < 2, f0_2 >> (16 * (g2 & 1)), f1_2 >> (16 * (g2 & 1))) & 0xFFFF
                    idx = q2 & 511
                    s7 = q2 >> 9
                    mask8 = s7 | ((_popc(s7) & 1) << 7)
                    gw = gl.load(GRID2_XS + idx * 2 + w2)
                else:
                    f0 = _lds_v(aw + so + ib * 4)
                    sg = _lds_v(aw + so + (8 + ib) * 4)
                    qhb = (_lds_v(aw + so + (16 + ib // 4) * 4) >> (8 * (ib % 4))) & 0xFF
                    scb = (_lds_v(ac + so + (18 + ib // 4) * 4) >> (8 * (ib % 4))) & 0xFF
                    f0_2 = zero2 + f0[:, None]
                    sg_2 = zero2 + sg[:, None]
                    qh_2 = zero2 + qhb[:, None]
                    idx = ((f0_2 >> (8 * g2)) & 0xFF) | (((qh_2 >> (2 * g2)) & 3) << 8)
                    mask8 = (sg_2 >> (8 * g2)) & 0xFF
                    gw = gl.load(GRID2_S + idx * 2 + w2)
                wq = _negate_bytes(gw, (mask8 >> nsh2i) & 0xF)
                wq_lo = gl.where(b2 < 4, wq, 0)
                wq_hi = gl.where(b2 >= 4, wq, 0)
                acc_lo = mma_v2(a8, _b_operand(wq_lo, db8), gl.zeros([BM, BN], gl.int32, mma))
                acc_hi = mma_v2(a8, _b_operand(wq_hi, db8), gl.zeros([BM, BN], gl.int32, mma))
                s_lo = scb & 0xF
                s_hi = (scb >> 4) & 0xF
                if E == 1:
                    acc_k = acc_k + acc_lo * (2 * s_lo + 1)[None, :] + acc_hi * (2 * s_hi + 1)[None, :]
                else:
                    sx = gl.load(sxrow + kb * 8 + ib, mask=ms_ok, other=0.0)
                    dl_lo = d * (2.0 * s_lo.to(gl.float32) + 1.0) * 0.125
                    dl_hi = d * (2.0 * s_hi.to(gl.float32) + 1.0) * 0.125
                    acc = acc + acc_lo.to(gl.float32) * (sx[:, None] * dl_lo[None, :]) + acc_hi.to(gl.float32) * (sx[:, None] * dl_hi[None, :])
            if E == 1:
                acc = acc + acc_k.to(gl.float32) * (sx[:, None] * (d * 0.125)[None, :])
        gl.barrier()
    ym = gl.arange(0, BM, layout=SliceLayout(1, mma))
    yn = gl.arange(0, BN, layout=SliceLayout(0, mma))
    omask = (ym[:, None] < M) & (yn[None, :] < n_valid)
    keep = smem.slice(0, 64).load(SliceLayout(0, mma))
    omask = omask & (keep[None, :] != 0x7FFFFFFF)
    if REGION == 0:   # 0046: only region 0's second allocation is otherwise untouched (regions 3 and 4 hold the A stages)
        keep2 = smem2.slice(0, 64).load(SliceLayout(0, mma))
        omask = omask & (keep2[None, :] != 0x7FFFFFFF)
    Yp = Y + split.to(gl.int64) * stride_yk
    gl.store(Yp + ym[:, None] * stride_ym + (col0 + yn)[None, :], acc.to(Y.dtype.element_ty), mask=omask)


@g.jit
def tiles_grouped_kernel(XQ, SX, SUMX, TILES, DESC, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
                         M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk,
                         BM: gl.constexpr, BN: gl.constexpr, STAGE_: gl.constexpr, TAB: gl.constexpr, E: gl.constexpr,
                         K0: gl.constexpr, K1: gl.constexpr, K2: gl.constexpr, K3: gl.constexpr,
                         AMODE: gl.constexpr, STAGES: gl.constexpr, GX: gl.constexpr, REGION: gl.constexpr, TABX: gl.constexpr, AOFF: gl.constexpr):
    smem_flat: gl.constexpr = SwizzledSharedLayout(4, 1, 1, [0])
    pid = gl.program_id(0)
    drow = DESC + pid * 16
    off = gl.load(drow + 0)
    tp = gl.load(drow + 1)
    col0 = gl.load(drow + 3)
    n_valid = gl.load(drow + 4)
    kb0 = gl.load(drow + 5)
    kb1 = gl.load(drow + 6)
    split = gl.load(drow + 7)
    # one flat region of 2 x STAGE_ words as two power-of-two allocations, consecutive (the copies and the
    # gathers address the region from its base; the two loads of the epilogue keep both allocations)
    if REGION == 1:
        # one [8192]-word region (32 KB, three blocks): the stages at 0 / 2304 / 4608, the A stages at 6144
        smem = gl.allocate_shared_memory(gl.int32, [8192], smem_flat)
        smem2 = smem
    elif REGION == 2:
        # the compact region (24 KB, four blocks): stages at 0 / 1792, the tables at 3584, the A stages in the second allocation
        smem = gl.allocate_shared_memory(gl.int32, [4096], smem_flat)
        smem2 = gl.allocate_shared_memory(gl.int32, [2048], smem_flat)
    elif REGION == 3:
        # 0046: stages at 0 / 2304 and the tables as in region 0, the 32-row A stages in the second allocation
        smem = gl.allocate_shared_memory(gl.int32, [8192], smem_flat)
        smem2 = gl.allocate_shared_memory(gl.int32, [4096], smem_flat)
    elif REGION == 4:
        # 0046 compact: stages at 0 / 1792 and the tables at 3584, the 32-row A stages in the second allocation
        smem = gl.allocate_shared_memory(gl.int32, [4096], smem_flat)
        smem2 = gl.allocate_shared_memory(gl.int32, [4096], smem_flat)
    else:
        smem = gl.allocate_shared_memory(gl.int32, [4096], smem_flat)
        smem2 = gl.allocate_shared_memory(gl.int32, [512], smem_flat)
    TILES32 = TILES.to(gl.pointer_type(gl.int32), bitcast=True)
    # the type is uniform per CTA: one branch, one compiled decode with its row words
    if tp == 21:
        _run(21, 28, E, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split, XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
             M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk, BM, BN, STAGE_, TAB, K0, K1, K2, K3, AMODE, STAGES, GX, REGION, TABX, AOFF)
    elif tp == 23:
        _run(23, 34, E, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split, XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
             M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk, BM, BN, STAGE_, TAB, K0, K1, K2, K3, AMODE, STAGES, GX, REGION, TABX, AOFF)
    elif tp == 12:
        _run(12, 36, E, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split, XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
             M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk, BM, BN, STAGE_, TAB, K0, K1, K2, K3, AMODE, STAGES, GX, REGION, TABX, AOFF)
    elif tp == 18:
        _run(18, 25, E, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split, XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
             M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk, BM, BN, STAGE_, TAB, K0, K1, K2, K3, AMODE, STAGES, GX, REGION, TABX, AOFF)
    elif tp == 16:
        _run(16, 17, E, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split, XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
             M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk, BM, BN, STAGE_, TAB, K0, K1, K2, K3, AMODE, STAGES, GX, REGION, TABX, AOFF)
    elif tp == 17:
        _run(17, 19, E, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split, XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
             M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk, BM, BN, STAGE_, TAB, K0, K1, K2, K3, AMODE, STAGES, GX, REGION, TABX, AOFF)
    else:
        _run(22, 21, E, smem, smem2, TILES32, off, kb0, kb1, col0, n_valid, split, XQ, SX, SUMX, Y, GRID_S, GRID_X, GRID2_XXS, GRID2_XS, GRID2_S,
             M, stride_xq, stride_sx, stride_sum, stride_ym, stride_yk, BM, BN, STAGE_, TAB, K0, K1, K2, K3, AMODE, STAGES, GX, REGION, TABX, AOFF)


def grouped_gemm(packed: torch.Tensor, meta, X: torch.Tensor, splitk: int | None = None, num_warps: int = 4,
                 quantized: tuple | None = None, out: torch.Tensor | None = None, e: bool = False,
                 maxnreg: int | None = None, amode: int | None = None, stages: int | None = None, gx: int = 1,
                 compact: bool | None = None, bm: int | None = None) -> torch.Tensor:
    """packed / meta from pack_layer; X bf16 [M <= 16, K]; returns bf16 [M, n_total] (into out if given).
    e=False: the per-32 arithmetic (quantized = quantize_activations(X, with_sums=True) or None);
    e=True: the per-256 sixth form (quantized = quantize_activations_256(X, with_sums=True) or None)."""
    assert X.is_cuda and X.dtype == torch.bfloat16 and X.dim() == 2
    M, K = X.shape
    nb = K // 256
    assert K % 256 == 0 and nb == meta["nb"] and M <= 32
    if bm is None:
        bm = 16 if M <= 16 else 32          # 0046: one launch of two mma row blocks above 16 rows
    assert bm in (16, 32) and M <= bm
    n_total = meta["n_total"]
    if quantized is not None and len(quantized) == 3:
        XQ, SX, SUMX = quantized
        assert SX.shape[1] == (nb if e else 8 * nb), "quantized does not match e"
    elif e:
        XQ, SX, SUMX = quantize_activations_256(X.contiguous(), with_sums=True)
    else:
        XQ, SX, SUMX = quantize_activations(X.contiguous(), with_sums=True)
    if splitk is None:
        splitk = grouped_split_for(sum(-(-n // 64) for _, _, _, n, _, _ in meta["shards"]), nb, m=M)
    splitk = max(1, min(splitk, nb))
    desc = descriptors(meta, splitk, X.device)
    dev = X.device
    if splitk == 1:
        Y = out if out is not None else torch.empty((M, n_total), dtype=X.dtype, device=dev)
        stride_ym, stride_yk = Y.stride(0), 0
    else:
        Y = torch.empty((splitk, M, n_total), dtype=torch.float32, device=dev)
        stride_ym, stride_yk = Y.stride(1), Y.stride(0)
    if amode is None:
        types = {wt for _, wt, _, _, _, _ in meta['shards']} if 'shards' in meta else set(meta.get('types', ()))
        # the M = 1 variant keeps the direct loads at five blocks; M != 1 stages the A tile, except the layers of IQ2
        # types only (pipeline-limited: the shared A reads cost them 4 %, the cell of 19:28)
        amode = 0 if (M == 1 or (types and types <= {16, 17, 22} and bm == 16)) else 2   # 0046: the direct-load A at bm 32 costs 246 registers
    if stages is None:
        stages = 2
    assert amode in (0, 1, 2) and stages in (2, 3) and not (amode == 2 and stages == 3)
    max_rw = meta.get("max_rw") or max(rw for _, _, rw, _, _, _ in meta["shards"])
    if compact is None:
        compact = False                      # the 24 KB region's fourth block costs the byte-heavy shapes 3-20 % (the cell of 19:35)
    assert not compact or (amode == 2 and max_rw <= 28)
    region = 2 if compact else (1 if (amode == 2 or stages == 3) else 0)
    if bm == 32 and amode == 2:
        region = 4 if max_rw <= 28 else 3     # 0046: the doubled A stages need the second allocation
    assert not (bm == 32 and amode == 2) or region in (3, 4), (bm, amode, region)
    R = REGIONS[region]
    tiles_grouped_kernel[(desc.shape[0],)](
        XQ, SX, SUMX, packed, desc, Y, grid32_iq3s(dev), grid32_iq3xxs(dev),
        grid_words(GGML_TYPE_IQ2_XXS, dev), grid_words(GGML_TYPE_IQ2_XS, dev), grid_words(GGML_TYPE_IQ2_S, dev),
        M, XQ.stride(0), SX.stride(0), SUMX.stride(0), stride_ym, stride_yk,
        BM=bm, BN=64, STAGE_=R["stage"], TAB=R["tab"], E=int(e), K0=T0, K1=T1, K2=T2, K3=T3,
        AMODE=amode, STAGES=stages, GX=int(gx), REGION=region, TABX=R["tabx"], AOFF=R["aoff"], num_warps=num_warps,
        **({"maxnreg": maxnreg} if maxnreg else {}))
    if splitk == 1:
        return Y
    return sum_partials(Y, out=out, dtype=X.dtype)
