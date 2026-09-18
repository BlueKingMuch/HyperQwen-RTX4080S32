"""Dispatch of the Gluon decode kernels: which GGUF types have one, and the
call. Used by the plugin's _fused_mul_mat_gguf for x.shape[0] <= GLUON_MAX_ROWS
(and up to twice that on row halves). Every
matmul type of the GSQ-RCO IQ3_S file is here; IQ1_M (one tensor) stays on
the batched MMVQ."""
from __future__ import annotations

import torch

from .iq2 import iq2s_gemm, iq2xs_gemm, iq2xxs_gemm
from .iq3s import iq3s_gemm
from .iq2_int8 import iq2s_int8_gemm, iq2xs_int8_gemm, iq2xxs_int8_gemm
from .iq3s_int8 import iq3s_int8_gemm, quantize_activations  # noqa: F401  (the quantiser and the helpers; 0040 quantises once per layer call)
from .iq3s_int8b import iq3s_int8b_gemm  # noqa: F401  (the third form, kept importable)
from .iq3s_int8c import iq3s_int8c_gemm
from .iq3s_int8t import iq3s_int8t_gemm, repack_iq3s_tiles  # noqa: F401  (0036: the tile-major form; the loader repacks)
from .iq3s_tile_dequant import dequantize_iq3s_tiles  # 0037: the prefill's dequantiser on the tiles
from .tiles import TILE_TYPES, dequantize_tiles, repack_tiles, tiles_gemm  # noqa: F401  (0038: every int8 type tile-major; the loader repacks)
from .tiles_grouped import grouped_gemm, grouped_split_for, prepare_grouped_layer, quantize_activations_256  # noqa: F401  (0042: one kernel for every tile-type layer; the loader packs)
from .iq3xxs_int8 import iq3xxs_int8_gemm
from .iq4xs_int8 import iq4xs_int8_gemm
from .q4k_int8 import q4k_int8_gemm
from .iq3xxs import iq3xxs_gemm
from .iq4xs import iq4xs_gemm
from .q2k import q2k_gemm
from .q4k import q4k_gemm

GGML_TYPE_Q2_K = 10
GGML_TYPE_Q4_K = 12
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
_KERNELS = {
    GGML_TYPE_IQ3_S: iq3s_gemm,
    GGML_TYPE_IQ4_XS: iq4xs_gemm,
    GGML_TYPE_IQ3_XXS: iq3xxs_gemm,
    GGML_TYPE_Q4_K: q4k_gemm,
    GGML_TYPE_IQ2_S: iq2s_gemm,
    GGML_TYPE_IQ2_XS: iq2xs_gemm,
    GGML_TYPE_IQ2_XXS: iq2xxs_gemm,
    GGML_TYPE_Q2_K: q2k_gemm,
}
# gguf-rco step 6h: the int8 form (mma.m16n8k32.s8, the activations quantised
# to int8 per 32 with an absmax scale, as the plugin's MMVQ does) for the types
# that have it; IQ3_S measured 1.38x the bf16 form at base clocks with the
# caches flushed (386 GB/s), the 0030 session's battery unchanged on it.
# 0031: IQ4_XS (the 16-entry table in four constant registers, two prmt per
# word), IQ3_XXS (grid words as int8, byte-wise sign negation) and Q4_K (the
# nibbles as unsigned int8, the min term from the activation group's sum).
# 0032: the IQ2 types (IQ2_XXS one mma per sub-block; IQ2_XS and IQ2_S,
# with a scale per 16, two mmas per sub-block on the halves of the B operand).
# Q2_K stays on the bf16 form (2 % of the Gluon time; a scale per 16 as well).
_INT8_KERNELS = {
    GGML_TYPE_IQ3_S: iq3s_int8c_gemm,     # 0035 the fourth form: the third's arithmetic, the A tile staged per k-block by cp.async, the grid table in shared memory
    GGML_TYPE_IQ4_XS: iq4xs_int8_gemm,
    GGML_TYPE_IQ3_XXS: iq3xxs_int8_gemm,
    GGML_TYPE_Q4_K: q4k_int8_gemm,
    GGML_TYPE_IQ2_S: iq2s_int8_gemm,
    GGML_TYPE_IQ2_XS: iq2xs_int8_gemm,
    GGML_TYPE_IQ2_XXS: iq2xxs_int8_gemm,
}
GLUON_TYPES = frozenset(_KERNELS)
GLUON_INT8_TYPES = frozenset(_INT8_KERNELS)
GLUON_MAX_ROWS = 16
GLUON_MAX_ROWS_GROUPED = 32   # 0046: the grouped kernel's one launch (bm 16 up to 16 rows, bm 32 above)
GLUON_ARRIVAL_ROWS = 128      # 0046: above one launch, 32-row blocks on the grouped kernel up to here, the dequant path beyond
GLUON_TILE_TYPES = frozenset(TILE_TYPES)     # 0038: the types the loader holds tile-major (every int8 type)


def gluon_mul_mat(x: torch.Tensor, weight: torch.Tensor, weight_type: int, n_out: int) -> torch.Tensor:
    """x bf16 [M <= 16, K]; weight the raw rows (any row stride); returns
    [M, n_out] in x's dtype."""
    fn = _INT8_KERNELS.get(weight_type) or _KERNELS.get(weight_type)
    if fn is None:
        raise NotImplementedError(f"no Gluon kernel for GGUF type {weight_type}")
    return fn(weight, x, n_out)


def gluon_mul_mat_iq3s_tiles(x: torch.Tensor, tiles: torch.Tensor, n_out: int) -> torch.Tensor:
    """0036 / 0037: x bf16 [M, K]; tiles the tile-major repack [n_tiles, nb, 64, 112]
    of an IQ3_S tensor's rows (repack_iq3s_tiles); returns [M, n_out] in x's
    dtype. Up to 16 rows the tile kernel, up to 32 in halves, above that the
    prefill path (the tile dequantiser and cuBLAS)."""
    if x.shape[0] <= GLUON_MAX_ROWS:
        return iq3s_int8t_gemm(tiles, x, n_out)
    if x.shape[0] <= 2 * GLUON_MAX_ROWS:
        return torch.cat([iq3s_int8t_gemm(tiles, x[:GLUON_MAX_ROWS], n_out), iq3s_int8t_gemm(tiles, x[GLUON_MAX_ROWS:], n_out)], dim=0)
    # 0037: above 32 rows the prefill path - dequantise from the tiles, cuBLAS
    w = dequantize_iq3s_tiles(tiles, n_out, x.shape[1], x.dtype)
    return x @ w.T


def gluon_mul_mat_tiles(x: torch.Tensor, tiles: torch.Tensor, n_out: int, weight_type: int) -> torch.Tensor:
    """0038: x bf16 [M, K]; tiles the type's tile-major repack (repack_tiles);
    returns [M, n_out] in x's dtype. Up to 16 rows the type's tile kernel, up
    to 32 in halves, above that the prefill path (the tiles dequantised - IQ3_S
    by its tile dequantiser, the others un-repacked for the CUDA one - and
    cuBLAS). The row-count dispatch lives here, inside the op (lesson 7b)."""
    if x.shape[0] <= GLUON_MAX_ROWS:
        return tiles_gemm(tiles, x, n_out, weight_type)
    if x.shape[0] <= 2 * GLUON_MAX_ROWS:
        return torch.cat([tiles_gemm(tiles, x[:GLUON_MAX_ROWS], n_out, weight_type), tiles_gemm(tiles, x[GLUON_MAX_ROWS:], n_out, weight_type)], dim=0)
    w = dequantize_tiles(tiles, n_out, x.shape[1], weight_type, x.dtype)
    return x @ w.T


def gluon_mul_mat_tiles_multi(x: torch.Tensor, tiles: list[torch.Tensor], n_outs: list[int], weight_types: list[int]) -> torch.Tensor:
    """0040: the shards of a merged layer in one call - x bf16 [M, K]; per shard its tile-major repack, its
    rows and its type; returns [M, sum(n_outs)] in x's dtype, the shards' results side by side. Up to 32 rows
    the activations are quantised once and every shard's tile kernel writes its column slice (up to 16 rows
    one launch per shard, up to 32 in row halves); above, the prefill path per shard (the dequantised tiles
    and cuBLAS, the GEMM as in the single-shard op, copied into the slice). The row-count dispatch lives
    here, inside the op (lesson 7b)."""
    M = int(x.shape[0])
    out = torch.empty((M, sum(n_outs)), dtype=x.dtype, device=x.device)
    col = 0
    if M <= 2 * GLUON_MAX_ROWS:
        xq = quantize_activations(x.contiguous(), with_sums=True)
        for t, n, wt in zip(tiles, n_outs, weight_types):
            view = out[:, col:col + n]
            if M <= GLUON_MAX_ROWS:
                tiles_gemm(t, x, n, wt, quantized=xq, out=view)
            else:
                lo = tuple(q[:GLUON_MAX_ROWS] for q in xq)
                hi = tuple(q[GLUON_MAX_ROWS:] for q in xq)
                tiles_gemm(t, x[:GLUON_MAX_ROWS], n, wt, quantized=lo, out=view[:GLUON_MAX_ROWS])
                tiles_gemm(t, x[GLUON_MAX_ROWS:], n, wt, quantized=hi, out=view[GLUON_MAX_ROWS:])
            col += n
        return out
    for t, n, wt in zip(tiles, n_outs, weight_types):
        w = dequantize_tiles(t, n, x.shape[1], wt, x.dtype)
        out[:, col:col + n].copy_(x @ w.T)
        col += n
    return out


def gluon_mul_mat_tiles_grouped(x: torch.Tensor, packed: torch.Tensor, descs: list[torch.Tensor], desc_splits: list[int],
                                tiles: list[torch.Tensor], n_outs: list[int], weight_types: list[int], nb: int, block_bytes: int) -> torch.Tensor:
    """0042: a layer's run of tile-type shards (one or many) in one call - x bf16 [M, K]; packed the run's tiles
    in one buffer (prepare_grouped_layer), descs the descriptor tables of desc_splits, tiles the shards' views
    into the buffer with their rows and types; returns [M, sum(n_outs)] in x's dtype, the shards' results side
    by side. Up to 32 rows (0046): the activations quantised once per 256 (with the int32 sums per 32 the K types'
    mins take), one launch of the grouped kernel - bm 16 up to 16 rows, bm 32 above - at the split of the table
    for that row count (precomputed at load for 1..32 rows: a split outside the tables would build one inside the
    compiled graph), the type uniform per CTA selecting the compiled decode, every tile writing its own columns,
    the partials summed in one fp32 kernel; 33-128 rows as 32-row blocks on the same kernel (the arrival step);
    above, the prefill path per shard (the dequantised tiles and cuBLAS, into the slice). The row-count dispatch
    lives here, inside the op (lesson 7b)."""
    M, K = int(x.shape[0]), int(x.shape[1])
    n_total = sum(n_outs)
    out = torch.empty((M, n_total), dtype=x.dtype, device=x.device)
    if M <= GLUON_ARRIVAL_ROWS:
        # 0046: up to 32 rows one launch (bm 16 up to 16 rows - 0044's launch to the byte - and bm 32 above: two mma
        # row blocks per decoded fragment, the weights streamed once); 33-128 rows (the arrival step: a new prompt
        # beside the running verify batches) as 32-row blocks on the same kernel, the weights streamed ceil(M / 32)
        # times instead of the dequant path's 114 GB; the split of the table for the block's row count
        n_tiles = sum(-(-n // 64) for n in n_outs)
        xq = quantize_activations_256(x.contiguous(), with_sums=True)
        for r0 in range(0, M, GLUON_MAX_ROWS_GROUPED):
            r1 = min(M, r0 + GLUON_MAX_ROWS_GROUPED)
            s = grouped_split_for(n_tiles, nb, m=r1 - r0, block_bytes=block_bytes)
            meta = {"nb": nb, "n_total": n_total, "desc": {s: descs[desc_splits.index(s)]}, "types": weight_types,
                    "max_rw": max(TILE_TYPES[wt][1] // 4 for wt in weight_types)}   # 0043: the launcher's variant by types
            grouped_gemm(packed, meta, x[r0:r1], splitk=s, quantized=tuple(q[r0:r1] for q in xq), out=out[r0:r1], e=True,
                         bm=16 if r1 - r0 <= 16 else 32)
        return out
    col = 0
    for t, n, wt in zip(tiles, n_outs, weight_types):
        w = dequantize_tiles(t, n, K, wt, x.dtype)
        out[:, col:col + n].copy_(x @ w.T)
        col += n
    return out
