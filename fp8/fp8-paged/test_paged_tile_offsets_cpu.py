"""CPU proof that the per-tile addressing equals the per-position form.

Two layers, no GPU:
  1. A Python model of both forms, exhaustive over every tile of several
     (BLOCK_SIZE, TILE_SIZE) pairs including 880/32 (the shipped geometry) and
     block sizes that are and are not multiples of the tile: same physical
     block, same slot, same K and V offsets for every position, and the same
     block-table entries read.
  2. --interpret: the real Triton helper, extracted from the candidate
     source, run under TRITON_INTERPRET=1 against the per-position form.

    python test_paged_tile_offsets_cpu.py
    TRITON_INTERPRET=1 python test_paged_tile_offsets_cpu.py --interpret --candidate candidate/triton_unified_attention.py
"""
import argparse
import math
import random

GEOMETRIES = [(880, 32), (880, 16), (32, 32), (64, 32), (16, 16), (1024, 32), (100, 32), (33, 32), (48, 16)]
S_K0, S_K1, S_V0, S_V1 = 1802240, 2048, 1802240, 2048


def per_position(table, pos, block, s0, s1):
    b = pos // block
    return table[b], pos % block, table[b] * s0 + (pos % block) * s1, b


def per_tile(table, j, tile, block, s0, s1):
    """Mirror of _paged_tile_offsets, one arm (K or V), returns lists + reads."""
    tile_start = j * tile
    blk0 = tile_start // block
    rem0 = tile_start - blk0 * block
    crosses = rem0 + tile > block
    phys0 = table[blk0]
    reads = {blk0}
    if crosses:
        phys1 = table[blk0 + 1]
        reads.add(blk0 + 1)
    else:
        phys1 = 0
    base0 = phys0 * s0 + rem0 * s1
    base1 = phys1 * s0 + (rem0 - block) * s1
    out = []
    for t in range(tile):
        wrap = t >= block - rem0
        phys = phys1 if wrap else phys0
        slot = (rem0 - block if wrap else rem0) + t
        off = (base1 if wrap else base0) + t * s1
        out.append((phys, slot, off))
    return out, reads


def model_check():
    rng = random.Random(19063)
    cases = 0
    for block, tile in GEOMETRIES:
        for n_blocks in (1, 2, 3, 7, 115):
            table = rng.sample(range(1, 1 << 20), n_blocks + 2)   # +2: padding entries as vLLM's table has
            positions = n_blocks * block
            for j in range(math.ceil(positions / tile) + 1):     # +1: one tile fully past the end
                new, reads = per_tile(table, j, tile, block, S_K0, S_K1)
                old_reads = set()
                for t in range(tile):
                    pos = j * tile + t
                    b = pos // block
                    if b >= len(table):
                        break
                    phys, slot, off, _ = per_position(table, pos, block, S_K0, S_K1)
                    assert new[t] == (phys, slot, off), (block, tile, n_blocks, j, t, new[t], (phys, slot, off))
                    old_reads.add(b)
                    cases += 1
                assert reads == old_reads, (block, tile, j, reads, old_reads)
    print(f"model: {cases} positions identical across {len(GEOMETRIES)} geometries")


def interpret_check(candidate):
    import os
    assert os.environ.get("TRITON_INTERPRET") == "1", "set TRITON_INTERPRET=1 before importing triton"
    import torch
    import triton
    import triton.language as tl
    src = open(candidate, encoding="utf8").read()
    pointer_variant = "def _paged_tile_ptrs(" in src
    start = src.index("@triton.jit\ndef _paged_tile_ptrs(" if pointer_variant else "@triton.jit\ndef _paged_tile_offsets(")
    end = src.index("\n\n\n", start)
    # The probe must live in the helper's own globals: a jitted function
    # resolves callees through its module globals, not closures.  The
    # pointer variant stores each position's pointer minus the cache base,
    # which is the same byte offset the offset variant returns.
    if pointer_variant:
        call = '''    phys, slot, k, v = _paged_tile_ptrs(
        table_ptr, k_cache, v_cache, tile_idx, offs_t, s_k0, s_k1, s_v0, s_v1, BLOCK_SIZE, TILE_SIZE)
    k = k.to(tl.int64) - k_cache.to(tl.int64)
    v = v.to(tl.int64) - v_cache.to(tl.int64)
'''
    else:
        call = '''    phys, slot, k, v = _paged_tile_offsets(
        table_ptr, tile_idx, offs_t, s_k0, s_k1, s_v0, s_v1, BLOCK_SIZE, TILE_SIZE)
'''
    probe_src = '''
@triton.jit
def probe(table_ptr, k_cache, v_cache, out_phys, out_slot, out_k, out_v, tile_idx,
          s_k0: tl.int64, s_k1: tl.int64, s_v0: tl.int64, s_v1: tl.int64,
          BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr):
    offs_t = tl.arange(0, TILE_SIZE)
''' + call + '''    tl.store(out_phys + offs_t, phys)
    tl.store(out_slot + offs_t, slot)
    tl.store(out_k + offs_t, k)
    tl.store(out_v + offs_t, v)
'''
    # The interpreter re-reads each jitted function's source through
    # ``inspect``, so the extracted helper and the probe go into a real
    # module file, in a temporary directory (never next to an installed tree).
    import importlib.util
    import tempfile
    module_path = os.path.join(tempfile.mkdtemp(prefix="paged_tile_offsets_"), "_paged_tile_offsets_probe.py")
    with open(module_path, "w", encoding="utf8") as f:
        f.write("import triton\nimport triton.language as tl\n\n\n" + src[start:end] + "\n" + probe_src)
    spec = importlib.util.spec_from_file_location("paged_tile_offsets_probe", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    probe = mod.probe

    rng = random.Random(7)
    cases = 0
    for block, tile in GEOMETRIES:
        n_blocks = 5
        table = rng.sample(range(1, 1 << 20), n_blocks + 2)
        t_table = torch.tensor(table, dtype=torch.int32)
        k_cache = torch.zeros(4096, dtype=torch.uint8)   # base pointers only; never dereferenced
        v_cache = torch.zeros(4096, dtype=torch.uint8)
        for j in range(math.ceil(n_blocks * block / tile)):
            outs = [torch.zeros(tile, dtype=d) for d in (torch.int64, torch.int32, torch.int64, torch.int64)]
            probe[(1,)](t_table, k_cache, v_cache, *outs, j, S_K0, S_K1, S_V0, S_V1, BLOCK_SIZE=block, TILE_SIZE=tile)
            for t in range(tile):
                pos = j * tile + t
                if pos // block >= len(table):
                    break
                phys, slot, off_k, _ = per_position(table, pos, block, S_K0, S_K1)
                _, _, off_v, _ = per_position(table, pos, block, S_V0, S_V1)
                got = (int(outs[0][t]), int(outs[1][t]), int(outs[2][t]), int(outs[3][t]))
                assert got == (phys, slot, off_k, off_v), (block, tile, j, t, got, (phys, slot, off_k, off_v))
                cases += 1
    print(f"interpreter: {cases} positions identical, triton {triton.__version__}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interpret", action="store_true")
    p.add_argument("--candidate", default="candidate/triton_unified_attention.py")
    a = p.parse_args()
    model_check()
    if a.interpret:
        interpret_check(a.candidate)
    print("PAGED_TILE_OFFSETS_CPU_PASS")
