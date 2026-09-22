"""The tile layouts of Q8_0 (8), Q3_K (11), Q5_K (13) and Q6_K (14) for the grouped kernel (tiles_grouped.py):
the bytes of a row's 256-weight block, the tile row, the stages a k-block takes, and the byte map from the
tile row to the block. The repack, the un-repack (the prefill path's raw rows for the plugin's CUDA
dequantiser) and the CPU gate all derive from the one map, so the layout is stated once.

    Q3_K   110 B: hmask[32] qs[64] scales[12] d      -> 112 B as it is, two bytes of padding (28 words, one stage)
    Q5_K   176 B: d dmin scales[12] qh[32] qs[128]   -> 176 B as it is (44 words, one stage of 2,816 words)
    Q6_K   210 B: ql[128] qh[64] scales[16] d        -> 2 x 108 B, one stage per half of 128 weights:
                                                         ql[64 h..] qh[32 h..] scales[8 h..] d, two bytes of padding (27 words)
    Q8_0   8 x 34 B (d qs[32])                       -> 2 x 136 B, one stage per four blocks: qs[32] x 4, d x 4 (34 words)

The k-block of a two-stage type is laid out stage-major ([2, 64, half bytes]) so that every stage is one
contiguous copy; the registry's [n_tiles, nb, 64, row_bytes] view holds the same bytes.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

GGML_TYPE_Q8_0, GGML_TYPE_Q3_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K = 8, 11, 13, 14
BLOCK_BYTES = {GGML_TYPE_Q8_0: 272, GGML_TYPE_Q3_K: 110, GGML_TYPE_Q5_K: 176, GGML_TYPE_Q6_K: 210}   # per 256 weights
ROW_BYTES = {GGML_TYPE_Q8_0: 272, GGML_TYPE_Q3_K: 112, GGML_TYPE_Q5_K: 176, GGML_TYPE_Q6_K: 216}
NST = {GGML_TYPE_Q8_0: 2, GGML_TYPE_Q3_K: 1, GGML_TYPE_Q5_K: 1, GGML_TYPE_Q6_K: 2}                   # stages per k-block
NEW_BIT = {GGML_TYPE_Q8_0: 1, GGML_TYPE_Q3_K: 2, GGML_TYPE_Q5_K: 4, GGML_TYPE_Q6_K: 8}               # the kernel's NEW mask
KQ_TYPES = frozenset(BLOCK_BYTES)


def tile_map(wt: int) -> list[int]:
    """Tile byte i (stage-major) -> the block byte it holds, -1 for padding."""
    if wt == GGML_TYPE_Q8_0:
        m = []
        for h in range(2):
            m += [(4 * h + j // 32) * 34 + 2 + j % 32 for j in range(128)]
            m += [(4 * h + j // 2) * 34 + j % 2 for j in range(8)]
        return m
    if wt == GGML_TYPE_Q3_K:
        return list(range(110)) + [-1, -1]
    if wt == GGML_TYPE_Q5_K:
        return list(range(176))
    if wt == GGML_TYPE_Q6_K:
        m = []
        for h in range(2):
            m += [64 * h + j for j in range(64)] + [128 + 32 * h + j for j in range(32)] + [192 + 8 * h + j for j in range(8)] + [208, 209, -1, -1]
        return m
    raise KeyError(wt)


def block_map(wt: int) -> list[int]:
    """Block byte k -> the first tile byte that holds it (the un-repack; Q6_K's d is in both halves, the first counts)."""
    inv = [-1] * BLOCK_BYTES[wt]
    for i, k in enumerate(tile_map(wt)):
        if k >= 0 and inv[k] < 0:
            inv[k] = i
    assert min(inv) >= 0, wt
    return inv


for _wt in KQ_TYPES:
    _m = tile_map(_wt)
    assert len(_m) == ROW_BYTES[_wt] and ROW_BYTES[_wt] % (4 * NST[_wt]) == 0, _wt
    assert sorted(set(k for k in _m if k >= 0)) == list(range(BLOCK_BYTES[_wt])), _wt

_MAPS: dict[tuple, torch.Tensor] = {}


def _map_tensor(wt: int, device, inverse: bool) -> torch.Tensor:
    key = (wt, str(device), inverse)
    t = _MAPS.get(key)
    if t is None:
        t = torch.tensor(block_map(wt) if inverse else tile_map(wt), dtype=torch.int32, device=device)
        _MAPS[key] = t
    return t


def repack(W: torch.Tensor, n_out: int, wt: int) -> torch.Tensor:
    """Raw rows uint8 [n_out, nb * block] -> tile-major uint8 [n_tiles, nb, 64, row_bytes] in the stage order,
    rows past n_out zero."""
    block, rb, nst = BLOCK_BYTES[wt], ROW_BYTES[wt], NST[wt]
    assert W.dtype == torch.uint8 and W.dim() == 2 and W.shape[0] == n_out and W.shape[1] % block == 0, (W.shape, wt)
    nb = W.shape[1] // block
    n_tiles = -(-n_out // 64)
    m = _map_tensor(wt, W.device, False).long()
    t = W.reshape(n_out, nb, block)[:, :, m.clamp(min=0)] * (m >= 0).to(torch.uint8)
    if n_tiles * 64 > n_out:
        t = torch.cat([t, torch.zeros((n_tiles * 64 - n_out, nb, rb), dtype=torch.uint8, device=W.device)], dim=0)
    return t.reshape(n_tiles, 64, nb, nst, rb // nst).permute(0, 2, 3, 1, 4).reshape(n_tiles, nb, 64, rb).contiguous()


def unrepack_torch(tiles: torch.Tensor, n_out: int, wt: int) -> torch.Tensor:
    """The tiles back to the raw rows uint8 [n_out, nb * block] (torch ops, any device: the reference)."""
    block, rb, nst = BLOCK_BYTES[wt], ROW_BYTES[wt], NST[wt]
    assert tiles.dtype == torch.uint8 and tiles.dim() == 4 and tiles.shape[2] == 64 and tiles.shape[3] == rb
    n_tiles, nb = int(tiles.shape[0]), int(tiles.shape[1])
    t = tiles.reshape(n_tiles, nb, nst, 64, rb // nst).permute(0, 3, 1, 2, 4).reshape(n_tiles * 64, nb, rb)[:n_out]
    inv = _map_tensor(wt, tiles.device, True).long()
    return t[:, :, inv].reshape(n_out, nb * block).contiguous()


@triton.jit
def _unrepack_kq_kernel(T, R, INV, n_out, nb, BLOCK: tl.constexpr, ROW_BYTES: tl.constexpr, HB: tl.constexpr, NP: tl.constexpr):
    """One program per (64-row tile, k-block): block byte k of row r <- the tile byte INV[k] in the stage-major chunk."""
    tile = tl.program_id(0)
    kb = tl.program_id(1)
    rows = tl.arange(0, 64)[:, None]
    k = tl.arange(0, NP)[None, :]
    km = k < BLOCK
    u = tl.load(INV + k, mask=km, other=0)
    src = T + ((tile * nb + kb) * 64).to(tl.int64) * ROW_BYTES + (u // HB) * (64 * HB) + rows * HB + (u % HB)
    r = tile * 64 + rows
    m = km & (r < n_out)
    v = tl.load(src, mask=m, other=0)
    tl.store(R + r.to(tl.int64) * (nb * BLOCK) + kb * BLOCK + k, v, mask=m)


def unrepack(tiles: torch.Tensor, n_out: int, wt: int) -> torch.Tensor:
    """The tiles back to the raw rows uint8 [n_out, nb * block] (the Triton kernel; bit-exact by construction)."""
    block, rb, nst = BLOCK_BYTES[wt], ROW_BYTES[wt], NST[wt]
    assert tiles.dtype == torch.uint8 and tiles.is_contiguous() and tiles.dim() == 4 and tiles.shape[2] == 64 and tiles.shape[3] == rb
    n_tiles, nb = int(tiles.shape[0]), int(tiles.shape[1])
    assert n_tiles == -(-n_out // 64)
    R = torch.empty((n_out, nb * block), dtype=torch.uint8, device=tiles.device)
    _unrepack_kq_kernel[(n_tiles, nb)](tiles, R, _map_tensor(wt, tiles.device, True), n_out, nb, BLOCK=block, ROW_BYTES=rb, HB=rb // nst,
                                       NP=triton.next_power_of_2(block), num_warps=4)
    return R


_PACKS: dict[tuple, tuple] = {}


def single_gemm(tiles: torch.Tensor, x: torch.Tensor, n_out: int, wt: int, quantized: tuple | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """The registry's launcher for one shard of the type (tiles_gemm's contract: x bf16 [M <= 16, K], quantized
    the per-32 tuple of quantize_activations or None): the grouped kernel on the shard alone, its pack cached
    by the tiles' storage. In serving every tile type goes through the layer's grouped launch instead."""
    from .tiles_grouped import grouped_gemm, pack_tiles

    key = (tiles.data_ptr(), tuple(tiles.shape), wt)
    ent = _PACKS.get(key)
    if ent is None:
        # the entry holds the tensor it was packed from: the key is that tensor's address, and a freed tensor's
        # address is handed to the next allocation of the shape, which would read this shard's pack as its own
        # (the entry already carries a copy of the bytes, so this costs the copy twice, never in serving - the
        # tiles of a loaded layer live as long as the process)
        ent = pack_tiles([(tiles, n_out, wt)]) + (tiles,)
        _PACKS[key] = ent
    packed, meta, _ = ent
    # `out` may be a column view of the caller's tensor; the partials' sum writes a contiguous target, so a strided view is
    # filled by a copy (splitk == 1 stores into the view directly)
    y = grouped_gemm(packed, meta, x, quantized=quantized, out=out if (out is None or out.is_contiguous()) else None, e=False)
    if out is not None and y is not out:
        out.copy_(y)
        return out
    return y
