"""The tile-major types (recipe step 0038): the registry of the type's repack,
decode kernel and prefill dequantisation, and the tile layouts back to raw
rows - one Triton kernel for every tile type, so the prefill path can hand the plugin's CUDA
dequantiser the bytes it was written for (bit-exact by construction; no
per-type dequantiser to verify). The tile row of a type is either the raw
block as it is (IQ4_XS 136, Q4_K 144: word-aligned already) or the block's
bytes after its 2-byte d followed by d, then zero padding to a word multiple
(IQ3_S 110 -> 112, IQ3_XXS 98 -> 100, IQ2_XXS 66 -> 68, IQ2_XS 74 -> 76,
IQ2_S 82 -> 84). One program per (64-row tile, k-block): 64 x ROW_BYTES bytes
in, 64 x BLOCK bytes out at their row positions.

    tiles = repack_tiles(W, n_out, weight_type)              # the loader, once
    y = tiles_gemm(tiles, x, n_out, weight_type)             # M <= 16
    w = dequantize_tiles(tiles, n_out, K, weight_type, dtype)  # the prefill's weights (IQ3_S: 0037's tile dequantiser; else un-repack + the CUDA one)
    raw = unrepack_tiles(tiles, n_out, weight_type)          # uint8 [n_out, nb * block]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# ggml type -> (block bytes, the tile row's bytes, d moved behind the block's other bytes)
TILE_TYPES = {
    21: (110, 112, True),    # IQ3_S (0036: qs, qh, signs, scales, d, 2 bytes of padding)
    23: (136, 136, False),   # IQ4_XS
    12: (144, 144, False),   # Q4_K
    18: (98, 100, True),     # IQ3_XXS
    16: (66, 68, True),      # IQ2_XXS
    17: (74, 76, True),      # IQ2_XS
    22: (82, 84, True),      # IQ2_S
}


@triton.jit
def _unrepack_kernel(T, R, n_out, nb, BLOCK: tl.constexpr, ROW_BYTES: tl.constexpr, D_LAST: tl.constexpr, NP: tl.constexpr):
    tile = tl.program_id(0)
    kb = tl.program_id(1)
    rows = tl.arange(0, 64)[:, None]
    j = tl.arange(0, NP)[None, :]
    if D_LAST:
        sj = tl.where(j < 2, BLOCK - 2 + j, j - 2)        # raw byte j of the block <- tile byte
    else:
        sj = j
    src = T + ((tile * nb + kb) * 64).to(tl.int64) * ROW_BYTES + rows * ROW_BYTES + sj
    r = tile * 64 + rows
    m = (j < BLOCK) & (r < n_out)
    v = tl.load(src + tl.zeros_like(sj), mask=m, other=0)
    tl.store(R + r.to(tl.int64) * (nb * BLOCK) + kb * BLOCK + j, v, mask=m)


def unrepack_tiles(tiles: torch.Tensor, n_out: int, weight_type: int) -> torch.Tensor:
    """tiles uint8 [n_tiles, nb, 64, row_bytes] of the type -> the raw rows uint8 [n_out, nb * block]."""
    block, row_bytes, d_last = TILE_TYPES[weight_type]
    assert tiles.dtype == torch.uint8 and tiles.is_contiguous() and tiles.dim() == 4 and tiles.shape[2] == 64 and tiles.shape[3] == row_bytes
    n_tiles, nb = int(tiles.shape[0]), int(tiles.shape[1])
    assert n_tiles == -(-n_out // 64)
    R = torch.empty((n_out, nb * block), dtype=torch.uint8, device=tiles.device)
    _unrepack_kernel[(n_tiles, nb)](tiles, R, n_out, nb, BLOCK=block, ROW_BYTES=row_bytes, D_LAST=d_last,
                                    NP=triton.next_power_of_2(block), num_warps=4)
    return R


def unrepack_tiles_torch(tiles: torch.Tensor, n_out: int, weight_type: int) -> torch.Tensor:
    """The same in torch ops (the reference; any device)."""
    block, row_bytes, d_last = TILE_TYPES[weight_type]
    assert tiles.dtype == torch.uint8 and tiles.dim() == 4 and tiles.shape[2] == 64 and tiles.shape[3] == row_bytes
    t = tiles.permute(0, 2, 1, 3).reshape(-1, tiles.shape[1], row_bytes)[:n_out]
    t = torch.cat([t[..., block - 2:block], t[..., :block - 2]], dim=-1) if d_last else t[..., :block]
    return t.reshape(n_out, -1).contiguous()


# ---- the registry: the type's repack (the loader, once) and decode kernel (M <= 16)
from .iq2 import GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS  # noqa: E402
from .iq2_int8t import iq2_int8t_gemm, repack_iq2_tiles  # noqa: E402
from .iq3s_int8t import iq3s_int8t_gemm, repack_iq3s_tiles  # noqa: E402
from .iq3s_int8te import iq3s_int8te_gemm  # noqa: E402  (0041: the IQ3_S decode on the tile sixth form)
from .iq3s_tile_dequant import dequantize_iq3s_tiles  # noqa: E402
from .iq3xxs_int8t import iq3xxs_int8t_gemm, repack_iq3xxs_tiles  # noqa: E402
from .iq4xs_int8t import iq4xs_int8t_gemm, repack_iq4xs_tiles  # noqa: E402
from .q4k_int8t import q4k_int8t_gemm, repack_q4k_tiles  # noqa: E402

GGML_TYPE_Q4_K = 12
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ4_XS = 23


def _iq2(t):
    return (lambda w, n: repack_iq2_tiles(w, n, t), lambda wt, x, n, **k: iq2_int8t_gemm(wt, x, n, t, **k))


_REPACK = {
    GGML_TYPE_IQ3_S: lambda w, n: repack_iq3s_tiles(w, n),
    GGML_TYPE_IQ4_XS: lambda w, n: repack_iq4xs_tiles(w, n, 136),      # unpadded (the 136 / 140 cell)
    GGML_TYPE_Q4_K: lambda w, n: repack_q4k_tiles(w, n, 144),
    GGML_TYPE_IQ3_XXS: lambda w, n: repack_iq3xxs_tiles(w, n, 100),
}
_GEMM = {
    GGML_TYPE_IQ3_S: lambda wt, x, n, quantized=None, out=None: iq3s_int8te_gemm(wt, x, n, out=out),   # 0041: per-256 activations, quantised in the launcher (a per-32 tuple is not its input)
    GGML_TYPE_IQ4_XS: iq4xs_int8t_gemm,
    GGML_TYPE_Q4_K: q4k_int8t_gemm,
    GGML_TYPE_IQ3_XXS: iq3xxs_int8t_gemm,
}
for _t in (GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S):
    _REPACK[_t], _GEMM[_t] = _iq2(_t)
assert set(_REPACK) == set(_GEMM) == set(TILE_TYPES)


def repack_tiles(W: torch.Tensor, n_out: int, weight_type: int) -> torch.Tensor:
    """The raw rows of the type (uint8 [n_out, nb * block]) -> its tile-major layout [n_tiles, nb, 64, row_bytes]."""
    return _REPACK[weight_type](W, n_out)


def tiles_gemm(tiles: torch.Tensor, x: torch.Tensor, n_out: int, weight_type: int,
               quantized: tuple | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """x bf16 [M <= 16, K] times the tiles' weights: [M, n_out] in x's dtype. 0040: the activations may come
    quantised (the (int8 X, scales, group sums) of quantize_activations, once per layer call) and the result
    may go into `out`, a [M, n_out] view of the caller's tensor (a merged layer's column slice)."""
    return _GEMM[weight_type](tiles, x, n_out, quantized=quantized, out=out)


def dequantize_tiles(tiles: torch.Tensor, n_out: int, K: int, weight_type: int, dtype=torch.bfloat16) -> torch.Tensor:
    """The tiles' weights dequantised to [n_out, K] for the prefill's cuBLAS GEMM: IQ3_S through 0037's tile
    dequantiser (one pass, bit-identical to the CUDA one), the other types un-repacked for the plugin's CUDA
    dequantiser on the bytes it was written for."""
    if weight_type == GGML_TYPE_IQ3_S:
        return dequantize_iq3s_tiles(tiles, n_out, K, dtype)
    from ... import ops

    return ops.ggml_dequantize(unrepack_tiles(tiles, n_out, weight_type), weight_type, n_out, K, dtype)
