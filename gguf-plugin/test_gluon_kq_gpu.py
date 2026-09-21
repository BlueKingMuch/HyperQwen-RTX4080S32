"""The GPU check of the K-quant / Q8_0 tile types on the grouped Gluon kernel (kq_tiles.py, tiles_grouped.py).
Run inside the image with the card free (the serving container stopped):

    docker run --rm --gpus all --entrypoint /app/venv/bin/python -e HOME=/cache -v hyperqwen-rtx4080s32_qwen-cache:/cache \\
        -v "$PWD/gguf-plugin:/work/gp:ro" hyperqwen-rtx4080s32:local /work/gp/test_gluon_kq_gpu.py [--types 8,11,13,14] [--time] [--big]

(before the image is rebuilt from this tree, also mount gguf-plugin/gluon over
/app/venv/lib/python3.12/site-packages/vllm_gguf_plugin/triton/gluon so the run sees these kernels)

Per type and shape, on random blocks (every bit pattern of a quantised field is valid): the tiles' round trip on
the GPU; the grouped launch at 1 / 4 / 8 / 16 / 17 / 32 rows in the serving form (per-256 activations, e=True)
and in the per-32 form (e=False) against the fp64 model of the int8 arithmetic (kq_ref.parse: the same integer
fields, exact dots, the scales after) - the model's own fields reproduce gguf.quants.dequantize bit for bit -
and against the fp64 product with the dequantised weights (the activation quantisation's error, for the
record); the launch variants of every region (amode 0 / 2, bm 16 / 32); determinism (the launch against itself,
NaN != NaN, every comparison counted); a mixed layer (IQ3_S + the type) bit-identical per shard to the single
shard at the same split. --time: CUDA-graph replays over copies of the packed layer that exceed the 64 MB L2,
median of 9, GB/s of the raw bytes, beside the plugin's batched MMVQ (the path these types took before) and Q4_K
on the same shape (the scaffold's rate). --big adds the head shape (248,320 x 5,120: the model on 2,048 rows)."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/app/venv/lib/python3.12/site-packages")
from gguf.constants import GGMLQuantizationType as GT  # noqa: E402
from gguf.quants import dequantize as gguf_dequantize  # noqa: E402
from vllm_gguf_plugin import ops  # noqa: E402
from vllm_gguf_plugin.triton.gluon import kq_ref, kq_tiles, tiles  # noqa: E402
from vllm_gguf_plugin.triton.gluon import tiles_grouped as tg  # noqa: E402
from vllm_gguf_plugin.triton.gluon.iq3s_int8 import quantize_activations  # noqa: E402

NAMES = {8: "Q8_0", 11: "Q3_K", 13: "Q5_K", 14: "Q6_K", 12: "Q4_K", 21: "IQ3_S"}
# the shapes these types take in the ByteShape GPU-5 file (n_out, K): attn_v / attn_k, attn_gate, ffn_gate, attn_output, the in_proj_ba pair
SHAPES = [(1024, 5120), (6144, 5120), (17408, 5120), (5120, 6144), (48, 5120)]
BIG = [(248320, 5120)]
FAILS = []
COMPARISONS = 0


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg, flush=True)


def fields_to_gpu(f, dev):
    g = {"G": f["G"], "q": torch.from_numpy(f["q"]).to(dev), "dg": torch.from_numpy(f["dg"]).to(dev)}
    g["mn"] = torch.from_numpy(f["mn"]).to(dev) if f["mn"] is not None else None
    g["dmin"] = torch.from_numpy(f["dmin"]).to(dev) if f["dmin"] is not None else None
    return g


def int8_model_gpu(g, xq, sx, sumx, per_256):
    """kq_ref.int8_model in torch fp64 on the device (the same arithmetic)."""
    q, dg, G = g["q"], g["dg"], g["G"]
    n, nb, _ = dg.shape
    M = xq.shape[0]
    per = 256 // G
    dots = torch.einsum("mkgp,nkgp->mnkg", xq.double().reshape(M, nb, G, per), q.double().reshape(n, nb, G, per))
    scaled = dots * dg.double()[None]
    if per_256:
        y = torch.einsum("mnk,mk->mn", scaled.sum(-1), sx.double())
    else:
        s32 = sx.double().reshape(M, nb, 8)
        s = s32.repeat_interleave(G // 8, dim=-1) if G > 8 else s32
        y = torch.einsum("mnkg,mkg->mn", scaled, s)
    if g["mn"] is not None:
        sm = sumx.double().reshape(M, nb, 8)
        mn, dm = g["mn"].double(), g["dmin"].double()
        if per_256:
            y = y - torch.einsum("mkg,nkg,nk,mk->mn", sm, mn, dm, sx.double())
        else:
            y = y - torch.einsum("mkg,nkg,nk->mn", sm, mn, dm)
    return y


def rel_err(y, ref):
    """max |y - ref| over the rows' max |ref| (the fp32 accumulation's relative error)."""
    global COMPARISONS
    COMPARISONS += y.numel()
    scale = ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    return float(((y.double() - ref).abs() / scale).amax())


def launch(packed, meta, X, n_tiles, nb, block_bytes, e, amode=None, bm=None, out_dtype=torch.float32, splitk=None):
    """One grouped launch (meta from pack_tiles: the descriptors are built and cached inside it per split)."""
    M = X.shape[0]
    if e:
        xq = tg.quantize_activations_256(X.contiguous(), with_sums=True)
    else:
        xq = quantize_activations(X.contiguous(), with_sums=True)
    s = splitk if splitk is not None else tg.grouped_split_for(n_tiles, nb, m=M, block_bytes=block_bytes)
    out = torch.empty((M, meta["n_total"]), dtype=out_dtype, device=X.device)
    tg.grouped_gemm(packed, meta, X, splitk=s, quantized=xq, out=out, e=e, amode=amode, bm=bm if bm is not None else (16 if M <= 16 else 32))
    return out, xq, s


def prepare(raw_np, n_out, wt, dev):
    """The loader's step on one shard: (raw rows, tiles, packed buffer, meta with shards, the loader's splits for 1..32 rows, nb, block bytes)."""
    W = torch.from_numpy(raw_np).to(dev)
    t = tiles.repack_tiles(W, n_out, wt)
    _, _, _, splits, nb, block_bytes = tg.prepare_grouped_layer([t], [n_out], [wt])
    packed, meta = tg.pack_tiles([(t, n_out, wt)])
    return W, t, packed, meta, splits, nb, block_bytes


def check_type(wt, n_out, K, dev, rng, rows_model=None):
    name = NAMES[wt]
    raw = kq_ref.random_rows(rng, n_out, K, wt)
    W, t, packed, meta, splits, nb, block_bytes = prepare(raw, n_out, wt, dev)
    n_tiles = -(-n_out // 64)
    # the tiles back to the raw rows on the GPU (the prefill path's un-repack)
    back = tiles.unrepack_tiles(t, n_out, wt)
    if not torch.equal(back, W):
        fail(f"{name} [{n_out}x{K}] un-repack (GPU) != raw rows")
    # the model's fields (a row subset for the big shapes), the dequantised reference
    rows = np.arange(n_out) if rows_model is None else rows_model
    f = kq_ref.parse(raw[rows], wt)
    w_ref = gguf_dequantize(raw[rows], GT(wt)).reshape(len(rows), K)
    if not np.array_equal(kq_ref.dequantize(f), w_ref):
        fail(f"{name} parse != gguf.quants.dequantize")
    g = fields_to_gpu(f, dev)
    w64 = torch.from_numpy(w_ref).to(dev).double()
    rows_t = torch.from_numpy(np.asarray(rows)).to(dev)
    worst = {}
    for M in (1, 4, 8, 16, 17, 32):
        X = (torch.randn(M, K, generator=torch.Generator(device=dev).manual_seed(100 + M), device=dev) * 0.7).to(torch.bfloat16)
        ref64 = X.double() @ w64.T
        for e in (True, False):
            y, xq, s = launch(packed, meta, X, n_tiles, nb, block_bytes, e)
            yy = y[:, rows_t]
            model = int8_model_gpu(g, xq[0], xq[1], xq[2], per_256=e)
            em = rel_err(yy, model)
            ea = rel_err(yy, ref64)
            key = f"e{int(e)}"
            worst[key] = max(worst.get(key, 0.0), em)
            worst[key + "_act"] = max(worst.get(key + "_act", 0.0), ea)
            if not np.isfinite(em) or em > 2e-5:
                fail(f"{name} [{n_out}x{K}] M={M} e={int(e)} split={s}: {em:.2e} from the int8 model")
            if e:
                y2, _, _ = launch(packed, meta, X, n_tiles, nb, block_bytes, e)
                if y.numel() == 0 or not torch.equal(y, y2) or bool(torch.isnan(y).any()):
                    fail(f"{name} [{n_out}x{K}] M={M}: not deterministic or NaN")
        # every region with the type: amode 0 (regions 0 / 5), amode 2 at bm 16 (1 / 6) and bm 32 (3 / 7); the splits of the loader's table
        if M in (1, 8):
            for amode, bm in ((0, 16), (2, 16), (2, 32)):
                y, xq, s = launch(packed, meta, X, n_tiles, nb, block_bytes, True, amode=amode, bm=bm)
                em = rel_err(y[:, rows_t], int8_model_gpu(g, xq[0], xq[1], xq[2], per_256=True))
                if not np.isfinite(em) or em > 2e-5:
                    fail(f"{name} [{n_out}x{K}] M={M} amode={amode} bm={bm}: {em:.2e} from the int8 model")
        if M == 8:
            # the registry's launcher (tiles_gemm, the per-32 form) into a column view of a wider tensor, at the table's split
            wide = torch.empty((M, n_out + 64), dtype=torch.bfloat16, device=dev)
            xq32 = quantize_activations(X.contiguous(), with_sums=True)
            tiles.tiles_gemm(t, X, n_out, wt, quantized=xq32, out=wide[:, 64:])
            y0, _, _ = launch(packed, meta, X, n_tiles, nb, block_bytes, False, out_dtype=torch.bfloat16)
            if not torch.equal(wide[:, 64:], y0):
                fail(f"{name} [{n_out}x{K}] tiles_gemm into a column view differs from the grouped launch")
            for s in sorted(set(splits) | {1, 2, 3, 5, min(8, nb)}):
                y, xq, _ = launch(packed, meta, X, n_tiles, nb, block_bytes, True, splitk=s)
                em = rel_err(y[:, rows_t], int8_model_gpu(g, xq[0], xq[1], xq[2], per_256=True))
                if not np.isfinite(em) or em > 2e-5:
                    fail(f"{name} [{n_out}x{K}] M=8 split={s}: {em:.2e} from the int8 model")
    print(f"{name:5} [{n_out:6}x{K:5}] tiles {tuple(t.shape)} splits {splits} | max rel err vs int8 model: e1 {worst['e1']:.1e} e0 {worst['e0']:.1e} | vs fp64 (activation quantisation): e1 {worst['e1_act']:.1e} e0 {worst['e0_act']:.1e}", flush=True)
    return raw, W, t, packed, meta, splits, nb, block_bytes


def check_mixed(wt, dev, rng):
    """IQ3_S (70 rows) + the type (130 rows) + the type (48 rows) in one launch: every shard's columns bit-identical to the
    shard alone at the same split."""
    K = 1024
    raw21 = rng.integers(0, 256, size=(70, K // 256 * 110), dtype=np.uint8)
    raw21.reshape(70, K // 256, 110)[:, :, :2] = (rng.uniform(0.25, 1.0, size=(70, K // 256)) * 2.0 ** -6).astype(np.float16).view(np.uint8).reshape(70, K // 256, 2)   # a finite d
    rawa = kq_ref.random_rows(rng, 130, K, wt)
    rawb = kq_ref.random_rows(rng, 48, K, wt)
    parts = [(torch.from_numpy(raw21).to(dev), 70, 21), (torch.from_numpy(rawa).to(dev), 130, wt), (torch.from_numpy(rawb).to(dev), 48, wt)]
    ts = [tiles.repack_tiles(W, n, t) for W, n, t in parts]
    packed, meta = tg.pack_tiles(list(zip(ts, [70, 130, 48], [21, wt, wt])))
    nb, bb = meta["nb"], 0
    for M in (1, 8, 32):
        X = (torch.randn(M, K, generator=torch.Generator(device=dev).manual_seed(7 + M), device=dev) * 0.7).to(torch.bfloat16)
        for s in (1, 2):
            y, xq, _ = launch(packed, meta, X, 5, nb, bb, True, splitk=s)
            col = 0
            for (W, n, t), tt in zip(parts, ts):
                p1, m1 = tg.pack_tiles([(tt, n, t)])
                y1, _, _ = launch(p1, m1, X, -(-n // 64), nb, bb, True, splitk=s)
                if not torch.equal(y[:, col:col + n], y1):
                    fail(f"mixed layer with {NAMES[wt]}: shard {NAMES[t]} ({n} rows) differs from the single shard at M={M} split={s}")
                col += n
    print(f"mixed IQ3_S + {NAMES[wt]} x 2: bit-identical per shard at M 1 / 8 / 32, splits 1 / 2", flush=True)


def time_launch(fn, n_copies, reps=9, iters=10):
    """fn(i) launches on copy i; a CUDA graph of n_copies launches, replayed; median of reps of the per-launch time."""
    for i in range(n_copies):
        fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_copies):
            fn(i)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / (iters * n_copies))
    return float(np.median(ts))


def timing(wt, n_out, K, dev, rng, M=8):
    name = NAMES[wt]
    raw = kq_ref.random_rows(rng, n_out, K, wt)
    n_bytes = raw.nbytes
    n_copies = max(1, -(-256 * 2 ** 20 // n_bytes))
    n_tiles = -(-n_out // 64)
    X = (torch.randn(M, K, device=dev) * 0.7).to(torch.bfloat16)
    xq = tg.quantize_activations_256(X, with_sums=True)
    packs = []
    for _ in range(n_copies):
        W, t, packed, meta, splits, nb, block_bytes = prepare(raw, n_out, wt, dev)
        s = tg.grouped_split_for(n_tiles, nb, m=M, block_bytes=block_bytes)
        packs.append((packed, meta, s, W))
    out = torch.empty((M, n_out), dtype=torch.bfloat16, device=dev)
    t_g = time_launch(lambda i: tg.grouped_gemm(packs[i][0], packs[i][1], X, splitk=packs[i][2], quantized=xq, out=out, e=True), n_copies)
    # the plugin's batched MMVQ on the raw rows (the path these types took before)
    t_m = time_launch(lambda i: ops.ggml_mul_mat_vec_a8(packs[i][3], X, wt, n_out), n_copies)
    # Q4_K on the same shape: the scaffold's rate for a uniform type
    raw4 = rng.integers(0, 256, size=(n_out, K // 256 * 144), dtype=np.uint8)
    packs4 = []
    for _ in range(max(1, -(-256 * 2 ** 20 // raw4.nbytes))):
        W4, t4, p4, m4, s4, nb4, bb4 = prepare(raw4, n_out, 12, dev)
        sp = tg.grouped_split_for(n_tiles, nb4, m=M, block_bytes=bb4)
        packs4.append((p4, m4, sp))
    t_4 = time_launch(lambda i: tg.grouped_gemm(packs4[i][0], packs4[i][1], X, splitk=packs4[i][2], quantized=xq, out=out, e=True), len(packs4))
    print(f"{name:5} [{n_out:6}x{K:5}] M={M} split {packs[0][2]}: grouped {t_g*1e3:7.1f} us {n_bytes/t_g/1e6:6.0f} GB/s | MMVQ {t_m*1e3:7.1f} us {n_bytes/t_m/1e6:6.0f} GB/s | Q4_K same shape {t_4*1e3:7.1f} us {raw4.nbytes/t_4/1e6:6.0f} GB/s  (copies {n_copies}, raw {n_bytes/2**20:.0f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="8,11,13,14")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--shapes", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.init()
    print(torch.cuda.get_device_name(), "| triton", __import__("triton").__version__, flush=True)
    types = [int(x) for x in a.types.split(",")]
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")] if a.shapes else [(130, 768), (1024, 5120), (6144, 5120), (5120, 6144), (48, 5120)]
    rng = np.random.default_rng(11)
    for wt in types:
        for n_out, K in shapes:
            check_type(wt, n_out, K, dev, rng)
        if a.big:
            rows = np.sort(np.concatenate([rng.choice(248320 - 64, 1984, replace=False), np.arange(248320 - 64, 248320)]))
            check_type(wt, 248320, 5120, dev, rng, rows_model=rows)
        check_mixed(wt, dev, rng)
    # the compiled variants' registers and spills per (BM, E, NEW, AMODE, STAGES, REGION) (the constexprs in the cache key's order)
    import re as _re

    rows = {}
    for dc in tg.tiles_grouped_kernel.device_caches.values():
        for key, k in dc[0].items():
            c = [int(v) for v in _re.findall(r"constexpr'?\"?, (-?\d+)", str(key))]
            if len(c) >= 16 and c[0] in (16, 32) and c[4] in (0, 1) and c[10] in (0, 1, 2) and c[11] in (2, 3) and 0 <= c[13] <= 7:
                bm, e, new, amode, stages, region = c[0], c[4], c[5], c[10], c[11], c[13]
                rows[(bm, e, new, amode, stages, region)] = (k.n_regs, k.n_spills, k.metadata.shared)
    print("variants compiled:", len(rows), flush=True)
    for (bm, e, new, amode, stages, region), (r, sp, sh) in sorted(rows.items()):
        print(f"  bm {bm:2} e {e} new {new:2} amode {amode} stages {stages} region {region}: regs {r:3} spills {sp:2} shared {sh}", flush=True)
    if a.time:
        for wt in types:
            for n_out, K in ([(1024, 5120), (6144, 5120), (17408, 5120), (5120, 6144)] + (BIG if a.big else [])):
                timing(wt, n_out, K, dev, rng)
    print(f"comparisons: {COMPARISONS}")
    if COMPARISONS == 0:
        fail("no comparison ran")
    print("FAILED" if FAILS else "OK", len(FAILS), "failures")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
