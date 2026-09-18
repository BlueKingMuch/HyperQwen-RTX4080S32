"""The IQ3_S dequantiser on the tile-major layout (for 0037: the prefill path
reads the tiles, so the raw rows can go), written to reproduce the plugin's
CUDA dequantiser bit for bit: dl = float(d) * (0.5 + s) * 0.5 (two
roundings in that order), y = dl * grid_byte, negated by the sign, cast to
the output dtype - the same three roundings as dequantize_block_iq3_s in
csrc/gguf/dequantize.cuh. One program per (row, k-block): 112 bytes in, 256
values out.

    from iq3s_tile_dequant import dequantize_iq3s_tiles
    W = dequantize_iq3s_tiles(tiles, n_out, K, torch.bfloat16)   # [n_out, K]

"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .iq3s import grid32  # the same 512 x 4-byte grid the decode kernels use


@triton.jit
def _dequant_iq3s_tile_kernel(T, Y, GRID32, n_out, nb, stride_y, KMASK: tl.constexpr):
    r = tl.program_id(0)
    kb = tl.program_id(1)
    tile = r // 64
    row = r % 64
    base = T + ((tile * nb + kb) * 64 + row) * 112
    # the row's fields: qs bytes 0-63, qh 64-71, signs 72-103, scales 104-107, d 108-109
    j = tl.arange(0, 256)
    ib = j // 32                     # sub-block 0..7
    il = (j % 32) // 8               # 0..3: which 8-value group of the sub-block
    jj = j % 8                       # 0..7 within the group
    half = jj // 4                   # grid1 (0) or grid2 (1)
    k4 = jj % 4                      # byte within the grid word
    qs = tl.load(base + 8 * ib + 2 * il + half).to(tl.int32)
    qh = tl.load(base + 64 + ib).to(tl.int32)
    idx = qs | ((qh << (8 - 2 * il - half)) & 256)
    gw = tl.load(GRID32 + idx)
    gbyte = (gw >> (8 * k4)) & 0xFF
    sgn = tl.load(base + 72 + 4 * ib + il).to(tl.int32)
    neg = (sgn >> jj) & 1
    sc = tl.load(base + 104 + ib // 2).to(tl.int32)
    s = (sc >> (4 * (ib % 2))) & 0xF
    dlo = tl.load(base + 108).to(tl.int32)
    dhi = tl.load(base + 109).to(tl.int32)
    d = ((dhi << 8) | dlo).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    # the CUDA dequantiser computes float(d) * (0.5 + s) * 0.5 * (4 v): scaling by
    # powers of two commutes with the rounding, so d * (1 + 2 s) * v (the decode
    # kernels' form, v the grid's odd magnitude) rounds to the same bits
    dl = d * (1.0 + 2.0 * s.to(tl.float32))
    y = dl * gbyte.to(tl.float32)
    y = tl.where(neg == 1, -y, y)
    tl.store(Y + r.to(tl.int64) * stride_y + kb * 256 + j, y.to(Y.dtype.element_ty), mask=r < n_out)


def dequantize_iq3s_tiles(tiles: torch.Tensor, n_out: int, K: int, dtype=torch.bfloat16) -> torch.Tensor:
    assert tiles.dtype == torch.uint8 and tiles.is_contiguous() and tiles.dim() == 4 and tiles.shape[2] == 64 and tiles.shape[3] == 112
    nb = K // 256
    assert tiles.shape[1] == nb and tiles.shape[0] * 64 >= n_out
    Y = torch.empty((n_out, K), dtype=dtype, device=tiles.device)
    _dequant_iq3s_tile_kernel[(tiles.shape[0] * 64, nb)](tiles, Y, grid32(tiles.device), n_out, nb, Y.stride(0), KMASK=0, num_warps=4)
    return Y
