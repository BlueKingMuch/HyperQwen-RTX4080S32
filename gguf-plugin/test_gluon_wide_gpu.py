"""The wide forms of the grouped kernel (0047): bm 64 (four warps, region 8) and bm 128 (eight warps, region 9)
over M-blocks at split 1 - the chunked prefill's GEMM on the tiles. Per type and shape: the wide launch against
the 32-row decode form on the same packed layer (the same arithmetic order: bit-identical), against an fp64 GEMM
on gguf.quants' dequantised weights with the kernel's own int8 activations (the fp32 accumulation's rounding),
NaN-free, deterministic; the M-block edges (33, 64, 65, 128, 129, 2048 rows); a mixed layer of three shards;
the compiled variants' registers, spills and shared memory. --time: the launch at 2,048 rows against cuBLAS bf16
(+ the tiles' dequantisation) and cuBLAS int8, and the 33-128-row step against the 32-row blocks (CUDA-graph
replays, median of 9). --time-only skips the checks.

  docker run --rm --gpus all --entrypoint /app/venv/bin/python \
    -v $PWD/gguf-plugin/gluon:/app/venv/lib/python3.12/site-packages/vllm_gguf_plugin/triton/gluon:ro \
    -v $PWD/gguf-plugin/test_gluon_wide_gpu.py:/t.py:ro <image> /t.py [--time] [--types 23,21] [--big]
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/app/venv/lib/python3.12/site-packages")
from gguf.constants import GGMLQuantizationType as GT  # noqa: E402
from gguf.quants import dequantize as gguf_dequantize  # noqa: E402
from vllm_gguf_plugin.triton.gluon import interface as gi  # noqa: E402
from vllm_gguf_plugin.triton.gluon import kq_ref, tiles  # noqa: E402
from vllm_gguf_plugin.triton.gluon import tiles_grouped as tg  # noqa: E402
from vllm_gguf_plugin.triton.gluon.tiles_grouped import STAGE_ROW_WORDS as STAGE_ROW_WORDS_  # noqa: E402

NAMES = {23: "IQ4_XS", 21: "IQ3_S", 12: "Q4_K", 18: "IQ3_XXS", 16: "IQ2_XXS", 17: "IQ2_XS", 22: "IQ2_S", 11: "Q3_K", 14: "Q6_K", 8: "Q8_0"}
GTYPE = {23: GT.IQ4_XS, 21: GT.IQ3_S, 12: GT.Q4_K, 18: GT.IQ3_XXS, 16: GT.IQ2_XXS, 17: GT.IQ2_XS, 22: GT.IQ2_S, 11: GT.Q3_K, 14: GT.Q6_K, 8: GT.Q8_0}
WIDE_TYPES = sorted(NAMES)          # Q5_K's 44-word rows take the wide stages (regions 5-7), not the wide forms
# (n_out, K): ffn_gate, attn_v / attn_k, ffn_down's K, the in_proj_ba pair (n_valid 48 of the tile)
SHAPES = [(17408, 5120), (1024, 5120), (1024, 17408), (48, 5120)]
TIME_SHAPES = [(17408, 5120), (10240, 5120), (5120, 17408), (6144, 5120)]
CHECK_M = [33, 64, 65, 128, 129, 2048]
FAILS = []
COMPARISONS = 0


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg, flush=True)


def random_raw(rng, n_out, K, wt):
    """Random valid rows of the type: every bit pattern of the quantised fields is valid; d (and Q4_K's dmin)
    small finite fp16 - the first field of every type here."""
    if wt in (8, 11, 13, 14):
        return kq_ref.random_rows(rng, n_out, K, wt)
    nb, block = K // 256, tiles.TILE_TYPES[wt][0]
    raw = rng.integers(0, 256, size=(n_out, nb, block), dtype=np.uint8)
    raw[:, :, :2] = (rng.uniform(0.25, 1.0, size=(n_out, nb)) * 2.0 ** -6).astype(np.float16).view(np.uint8).reshape(n_out, nb, 2)
    if wt == 12:
        raw[:, :, 2:4] = (rng.uniform(0.25, 1.0, size=(n_out, nb)) * 2.0 ** -8).astype(np.float16).view(np.uint8).reshape(n_out, nb, 2)
    return raw.reshape(n_out, nb * block)


def rel_err(y, ref):
    """max |y - ref| over the rows' max |ref|."""
    global COMPARISONS
    COMPARISONS += y.numel()
    scale = ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    return float(((y.double() - ref.double()).abs() / scale).amax())


def decode_form(packed, meta, X, xq, out):
    """The 32-row form over row blocks at split 1 (the reference arithmetic)."""
    XQ, SX, SUMX = xq
    M = X.shape[0]
    for i in range(0, M, 32):
        j = min(M, i + 32)
        tg.grouped_gemm(packed, meta, X[i:j], splitk=1, quantized=(XQ[i:j], SX[i:j], SUMX[i:j]), out=out[i:j], e=True, bm=32)
    return out


def check_layer(name, packed, meta, raws, dev, ms, bms):
    """raws: the shards' raw rows (for the fp64 reference), in the packed order."""
    K = meta["nb"] * 256
    n_total = meta["n_total"]
    W = torch.cat([torch.from_numpy(gguf_dequantize(raw, GTYPE[wt]).astype(np.float64)) for raw, wt in raws], dim=0).to(dev)   # [sum n_out, K]
    cols = torch.cat([torch.arange(c0, c0 + n) for (_, _, _, n, _, c0) in meta["shards"]])
    for M in ms:
        X = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        xq = tg.quantize_activations_256(X, with_sums=True)
        XQ, SX, SUMX = xq
        ref = decode_form(packed, meta, X, xq, torch.empty((M, n_total), dtype=torch.float32, device=dev))
        Xh = XQ.double() * SX.double().repeat_interleave(256, dim=1)
        fp = Xh @ W.T
        for bm in bms:
            out = torch.empty((M, n_total), dtype=torch.float32, device=dev)
            tg.grouped_gemm(packed, meta, X, splitk=1, quantized=xq, out=out, e=True, bm=bm)
            out2 = torch.empty_like(out)
            tg.grouped_gemm(packed, meta, X, splitk=1, quantized=xq, out=out2, e=True, bm=bm)
            torch.cuda.synchronize()
            neq, nan, det = int((out != ref).sum()), int(out.isnan().sum()), int((out != out2).sum())
            e = rel_err(out[:, cols], fp)
            print(f"{name:28} M {M:5} bm {bm:3} | vs 32-row form: {neq} of {out.numel()} differ, nan {nan}, rerun differs {det} | vs fp64 GEMM (gguf.quants W, int8 X): rel {e:.1e}", flush=True)
            if neq or nan or det or not np.isfinite(e) or e > 1e-5:
                fail(f"{name} M {M} bm {bm}: differ {neq} nan {nan} rerun {det} rel {e:.1e}")


def check_type(wt, n_out, K, dev, rng):
    raw = random_raw(rng, n_out, K, wt)
    t = tiles.repack_tiles(torch.from_numpy(raw).to(dev), n_out, wt)
    packed, meta = tg.pack_tiles([(t, n_out, wt)])
    big = n_out * K >= 17408 * 5120
    ms = CHECK_M if big else [33, 64, 65, 129]
    check_layer(f"{NAMES[wt]} [{n_out}x{K}]", packed, meta, [(raw, wt)], dev, ms, (64, 128))


def check_mixed(dev, rng):
    """Three shards (IQ3_S 70 rows, IQ4_XS 130, Q6_K 48) in one layer: the wide forms on the packed layer."""
    K = 1024
    parts = [(21, 70), (23, 130), (14, 48)]
    raws = [(random_raw(rng, n, K, wt), wt) for wt, n in parts]
    ts = [tiles.repack_tiles(torch.from_numpy(raw).to(dev), n, wt) for (raw, wt), (_, n) in zip(raws, parts)]
    packed, meta = tg.pack_tiles([(t, n, wt) for t, (wt, n) in zip(ts, parts)])
    check_layer("mixed IQ3_S+IQ4_XS+Q6_K", packed, meta, raws, dev, [33, 100, 129, 300], (64, 128))


def check_dispatch(dev, rng):
    """The op above 128 rows (gluon_mul_mat_tiles_grouped, the loader's tables): a run of wide-form shards must
    come out of the wide launch bit-identical to the 32-row form, and a run holding one of GLUON_DEQUANT_TYPES
    must come out of the dequant path (bf16 cuBLAS on the dequantised tiles, so only close)."""
    K = 1280
    for parts, wide in (([(23, 130), (21, 192)], True), ([(23, 130), (12, 192)], False)):
        raws = [(random_raw(rng, n, K, wt), wt) for wt, n in parts]
        ts = [tiles.repack_tiles(torch.from_numpy(raw).to(dev), n, wt) for (raw, wt), (_, n) in zip(raws, parts)]
        packed, views, descs, splits, nb, block_bytes = tg.prepare_grouped_layer(ts, [n for _, n in parts], [wt for wt, _ in parts])
        if 1 not in splits:
            fail("the loader's tables carry no split 1")
        W = torch.cat([torch.from_numpy(gguf_dequantize(raw, GTYPE[wt]).astype(np.float64)) for raw, wt in raws], dim=0).to(dev)
        name = " + ".join(NAMES[wt] for wt, _ in parts)
        for M in (129, 300):
            X = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            y = gi.gluon_mul_mat_tiles_grouped(X, packed, descs, splits, views, [n for _, n in parts], [wt for wt, _ in parts], nb, block_bytes)
            xq = tg.quantize_activations_256(X, with_sums=True)
            ref = decode_form(packed, {"nb": nb, "n_total": sum(n for _, n in parts), "desc": {1: descs[splits.index(1)]},
                                       "types": [wt for wt, _ in parts], "max_rw": max(STAGE_ROW_WORDS_[wt] for wt, _ in parts)},
                              X, xq, torch.empty_like(y))
            fp = (xq[0].double() * xq[1].double().repeat_interleave(256, dim=1)) @ W.T
            torch.cuda.synchronize()
            neq = int((y != ref).sum())
            e = rel_err(y, fp)
            ok = (neq == 0) if wide else (e < 5e-2)
            print(f"dispatch {name:18} M {M:4} | {'wide form' if wide else 'dequant path'}: differs from the 32-row form in {neq} of {y.numel()}, rel to fp64 {e:.1e}", flush=True)
            if not ok or not np.isfinite(e):
                fail(f"dispatch {name} M {M}: differs {neq}, rel {e:.1e}")
    # Q5_K's 44-word rows do not fit the wide stages: the launcher must refuse the form, not launch a wrong one
    raw = random_raw(rng, 128, K, 13)
    t5 = tiles.repack_tiles(torch.from_numpy(raw).to(dev), 128, 13)
    p5, m5 = tg.pack_tiles([(t5, 128, 13)])
    X = (torch.randn(200, K, device=dev) * 0.5).to(torch.bfloat16)
    try:
        tg.grouped_gemm(p5, m5, X, splitk=1, e=True, bm=128)
        fail("Q5_K at bm 128 launched instead of being refused")
    except AssertionError:
        print("dispatch Q5_K              bm 128 | refused by the launcher (44-word rows)", flush=True)
    if 13 not in gi.GLUON_DEQUANT_TYPES or 12 not in gi.GLUON_DEQUANT_TYPES:
        fail("GLUON_DEQUANT_TYPES does not hold Q4_K and Q5_K")


def ev_time(fn, iters=10, reps=9):
    """The GPU time of fn per call: iters calls captured in one CUDA graph (the launch overhead of the host out of
    the measurement - a sub-millisecond launch timed around a Python call reads the host), replays timed with
    events, the median of reps."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters)
    return float(np.median(ts))


def timing(wt, n_out, K, dev, rng, M=2048):
    raw = random_raw(rng, n_out, K, wt)
    t = tiles.repack_tiles(torch.from_numpy(raw).to(dev), n_out, wt)
    packed, meta = tg.pack_tiles([(t, n_out, wt)])
    X = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    xq = tg.quantize_activations_256(X, with_sums=True)
    out = torch.empty((M, meta["n_total"]), dtype=torch.bfloat16, device=dev)
    Wd = tiles.dequantize_tiles(t, n_out, K, wt, torch.bfloat16)
    W8 = torch.randint(-127, 128, (n_out, K), dtype=torch.int8, device=dev)
    ops = 2.0 * M * n_out * K
    tw = ev_time(lambda: tg.grouped_gemm(packed, meta, X, splitk=1, quantized=xq, out=out, e=True, bm=128))
    tq = ev_time(lambda: tg.quantize_activations_256(X, with_sums=True))
    tb = ev_time(lambda: X @ Wd.T)
    td = ev_time(lambda: tiles.dequantize_tiles(t, n_out, K, wt, torch.bfloat16))
    ti = ev_time(lambda: torch._int_mm(xq[0], W8.t()))
    print(f"{NAMES[wt]:7} [{n_out:6}x{K:5}] M {M} | bm 128 {tw:7.3f} ms = {ops / tw / 1e9:6.1f} TOPS (quantise {tq:.3f}) | cuBLAS bf16 {tb:7.3f} ms = {ops / tb / 1e9:6.1f} (dequantise {td:.3f}) | cuBLAS int8 {ti:7.3f} ms = {ops / ti / 1e9:6.1f} | bf16/bm128 {tb / tw:.2f}x, (bf16+deq)/(bm128+quant) {(tb + td) / (tw + tq):.2f}x", flush=True)
    # the 33-128-row step: the 32-row blocks at the tables' split against one wide launch
    n_tiles = -(-n_out // 64)
    block_bytes = int(round(packed.numel() / (meta["n_total"] * meta["nb"])))
    for Ms, bm in ((64, 64), (64, 128), (128, 64), (128, 128)):
        Xs = X[:Ms]
        xqs = tuple(q[:Ms] for q in xq)
        outs = out[:Ms]
        s = tg.grouped_split_for(n_tiles, meta["nb"], m=32, block_bytes=block_bytes)
        tg.descriptors(meta, s, dev)

        def blocks():
            for r0 in range(0, Ms, 32):
                tg.grouped_gemm(packed, meta, Xs[r0:r0 + 32], splitk=s, quantized=tuple(q[r0:r0 + 32] for q in xqs), out=outs[r0:r0 + 32], e=True, bm=32)
        t32 = ev_time(blocks)
        tw_ = ev_time(lambda: tg.grouped_gemm(packed, meta, Xs, splitk=1, quantized=xqs, out=outs, e=True, bm=bm))
        print(f"{NAMES[wt]:7} [{n_out:6}x{K:5}] M {Ms:4} | 32-row blocks (split {s}) {t32:7.3f} ms | bm {bm:3} one launch {tw_:7.3f} ms | {t32 / tw_:.2f}x", flush=True)


def variants():
    rows = {}
    for dc in tg.tiles_grouped_kernel.device_caches.values():
        for key, k in dc[0].items():
            c = [int(v) for v in re.findall(r"constexpr'?\"?, (-?\d+)", str(key))]
            # BM, BN, STAGE_, TAB, E, NEW, NW, K0..K3, AMODE, STAGES, GX, REGION, TABX, AOFF
            if len(c) >= 17 and c[0] in (16, 32, 64, 128) and c[4] in (0, 1) and c[11] in (0, 1, 2) and c[12] in (2, 3) and 0 <= c[14] <= 9:
                rows[(c[0], c[4], c[5], c[6], c[11], c[12], c[14])] = (k.n_regs, k.n_spills, k.metadata.shared)
    print("variants compiled:", len(rows), flush=True)
    for (bm, e, new, nw, amode, stages, region), (r, sp, sh) in sorted(rows.items()):
        print(f"  bm {bm:3} e {e} new {new:2} warps {nw} amode {amode} stages {stages} region {region}: regs {r:3} spills {sp:2} shared {sh}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default=",".join(str(t) for t in WIDE_TYPES))
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--big", action="store_true", help="the head shape 248320 x 5120 (Q6_K) in the timing")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0), "| triton", __import__("triton").__version__, flush=True)
    rng = np.random.default_rng(a.seed)
    types = [int(v) for v in a.types.split(",")]
    if not a.time_only:
        for wt in types:
            for n_out, K in SHAPES:
                check_type(wt, n_out, K, dev, rng)
        check_mixed(dev, rng)
        check_dispatch(dev, rng)
        variants()
    if a.time or a.time_only:
        for wt in types:
            for n_out, K in TIME_SHAPES:
                timing(wt, n_out, K, dev, rng)
        if a.big:
            timing(14, 248320, 5120, dev, rng)
    print(f"comparisons: {COMPARISONS}")
    if COMPARISONS == 0 and not a.time_only:
        fail("no comparison ran")
    print("FAILED" if FAILS else "OK", len(FAILS), "failures")
    sys.exit(1 if FAILS else 0)
