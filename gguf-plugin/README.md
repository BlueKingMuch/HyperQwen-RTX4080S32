# The out-of-tree GGUF plugin, pinned and patched

The target's non-uniform GSQ-RCO quantizations exist only as GGUF, and vLLM
reads GGUF through an out-of-tree plugin. This directory is what it takes to get
that plugin to serve them: the plugin fetched at a pinned commit, ten patches,
twenty-six Gluon decode kernels (eleven tile types on the grouped one), a CPU gate
that says the result installed, and a GPU check of the four K-quant / Q8_0 tile types.

**None of the plugin's source is carried here.** `install.sh` fetches the
archive and checks its sha256, so what this repo holds is only what is ours. The
same shape the Dockerfile already uses for vLLM itself -- pin the upstream, keep
the changes as a series -- and the same shape as `kvarn/`, one directory beside
the thing it installs.

[← back to the main README](../README.md) · [the vLLM series](../PATCHES.md)

## Running it

```
bash gguf-plugin/install.sh --check    # verify what is carried here; no network
bash gguf-plugin/install.sh            # fetch, patch, build, install
python3 gguf-plugin/test_gguf_rco_cpu.py
```

`gguf-plugin/test_gluon_kq_gpu.py` and `gguf-plugin/test_gluon_wide_gpu.py` run on the
card (their docstrings have the `docker run`).

The Dockerfile runs it after `kvarn/install.sh` and `fp8/install.sh`, so every
image built from this repo has the plugin. It costs build time -- the extension
is compiled for two architectures -- and nothing at run time until a model path
ends in `.gguf`: `_is_gguf_model()` is the whole of the plugin's claim on a
model, and it reads the path.

`TORCH_CUDA_ARCH_LIST` defaults to `8.9;12.0`. sm_89 is this card; sm_120 is
there because SASS is not forward-compatible across generations and only PTX is,
so a Blackwell clone of this recipe would otherwise JIT every kernel on first
use. Override it for a different card.

## The three hash gates

They answer different questions, which is why there are three:

| gate | what it says |
|---|---|
| `SHA256SUMS` | what this repo carries is what was reviewed |
| the archive's sha256 | upstream's bytes at the pin are the bytes being patched |
| before / after, six files | the files the series touches entered and left in the exact states it was cut for |

The third is the one worth having. A patch that lands somewhere plausible but
wrong still applies; it fails here instead of in a kernel.

What has actually been run, on a pristine fetch of the pin:

- archive sha256 matches
- the six before-hashes match
- all ten patches apply with `--fuzz 0`
- the six after-hashes match
- the batched-dispatch marker is present

## The pin

`vllm-project/vllm-gguf-plugin` at `d4c1f0d082fc`, archive sha256
`c225ff0a282e…`. Apache-2.0; the ten patches in `patches/` are derivative works
of it and carry that licence with them. The Gluon kernels in `gluon/` are ours.

The sha256 protects integrity, not availability: if that archive stops being
reachable, `install.sh` cannot run. Vendoring the source instead would trade
that for carrying someone else's tree in this repo, which is the worse trade
while the archive is up.

## The series

`series` is the apply order and the single source of truth for it; `install.sh`
fails if it and `patches/` disagree in either direction. The Gluon modules are
copied into `vllm_gguf_plugin/triton/gluon/` immediately before
`gguf-gluon-dispatch.patch`, which is the patch that wires them into the
plugin's matmul — they are files, not a patch, so they are not in `series`.

| patch | what |
|---|---|
| gguf-loader-draft-bridge | a non-GGUF model — the safetensors draft — keeps its own architectures and goes to vLLM's default loader with its own quantization config, instead of being refused |
| gguf-mmvq-batched | the decode kernel batched: a row's weights are read once per step for up to 16 vectors instead of once per vector, the dispatch capped at 16 rows |
| gguf-gluon-dispatch | the Gluon decode kernels dispatched from the plugin's matmul for up to 16 rows of bf16 activations, for the GGUF types that have one |
| gguf-iq3s-grid-top-magnitude | the CUDA IQ3_S dequantiser and the MMVQ dot read the format's pre-2024-03 grid, whose top magnitude is 15.5 where the format says 15 |
| gguf-prefill-dispatch | prefill through dequantise + cuBLAS for every type instead of the plugin's MMQ; the Gluon path up to 32 rows on row halves |
| gguf-shard-tensors | a merged layer's shard goes in as its strided view, with no per-step copy |
| gguf-iq3s-tiles | the IQ3_S tile-major repack with 16-byte copies, the grid table and the A tile in shared memory |
| gguf-iq3s-tiles-only | the tiled path narrowed to where it wins |
| gguf-tiles-all-types | the tile-major form for the remaining int8 types |
| gguf-tiles-grouped | one kernel and one launch for a layer of mixed-type shards, with the 16/32-row choice at launch time |

The `gguf-rco step 6a` … `6h` markers the patches insert into the plugin's source
are what `test_gguf_rco_cpu.py` greps for. They are load-bearing, and the patch
files are content-hashed, so neither is edited to make them read differently.

## The K-quant and Q8_0 tile types

Q3_K (11), Q5_K (13), Q6_K (14) and Q8_0 (8) are tile types of the grouped kernel:
the layouts in `gluon/kq_tiles.py` (one byte map per type, from which the repack, the
un-repack for the prefill path's CUDA dequantiser and the CPU gate derive), the
decode branches in `gluon/tiles_grouped.py`. No row form and no separate tile
kernel; the loader repacks them as it does the seven others, `linear.py` is
untouched. A branch is compiled into a launch only when the layer holds the type
(the `NEW` mask); a layer without them compiles to the seven-type kernel.

| type | block bytes per 256 weights | tile row | stages per k-block | stage row words | decode |
|---|---|---|---|---|---|
| Q3_K | 110 | 112, the block as it is (d is its last field), 2 bytes of padding | 1 | 28 | 2-bit fields, −4 where the hmask bit is clear, 16 six-bit scales −32, two mmas per sub-block |
| Q5_K | 176 | 176, the block as it is | 1 | 44, the 2,816-word stages (regions 5–7, the shapes of 0 / 1 / 3) | Q4_K's decode with bit 4 from the qh plane, the min term from the activation sums |
| Q6_K | 210 | 2 × 108, the block's halves: ql, qh, scales, d, 2 bytes of padding | 2 | 27 | nibble and two qh bits −32 (signed), int8 scales per 16, two mmas per sub-block |
| Q8_0 | 272, eight blocks of 34 | 2 × 136, four blocks each: qs × 4, d × 4 | 2 | 34 | none; the fp16 d per 32 applied in fp32 per sub-block |

Registers per launch variant (sm_89, `cuobjdump -res-usage` on a CPU compile of the
variants; spills 0 everywhere, the bm-32 variants 8 bytes of stack):

| variant | NEW = 0 | Q3_K | Q5_K | Q6_K | Q8_0 | all four |
|---|---|---|---|---|---|---|
| M = 1: amode 0, bm 16, region 0 (Q5_K: 5) | 126 | 126 | 159 | 126 | 126 | |
| M = 2–16: amode 2, bm 16, region 1 (6) | 127 | 127 | 127 | 127 | 127 | 127 |
| M = 17–32: amode 2, bm 32, region 3 (7) | 168 | 168 | 168 | 168 | | 168 |

`test_gluon_kq_gpu.py` is the GPU check (the card free, the command in its docstring):
random blocks of every type at 130 × 768, 1,024 × 5,120, 6,144 × 5,120, 5,120 × 6,144,
48 × 5,120 and 248,320 × 5,120 (the shapes these types take in
`byteshape/Qwen3.8-27B-GGUF`'s GPU-5 file, where they hold 12.9 % of the bytes),
1 / 4 / 8 / 16 / 17 / 32 rows, both arithmetic forms, every region and every split
of the loader's table, against the fp64 model of the int8 arithmetic whose fields
reproduce `gguf.quants.dequantize` bit for bit: max relative error 6.4e-7 (per-256
form) and 9.5e-7 (per-32 form) over 22,509,656 comparisons; deterministic; a mixed
layer of IQ3_S and the type bit-identical per shard to the single shard; the
un-repack round trip on the GPU exact.

Per launch at 8 rows (the per-256 form, the split of the table, CUDA-graph replays
over copies of the layer exceeding the 64 MB L2, median of 9 samples; GB/s of the
raw block bytes; MMVQ the plugin's batched `ggml_mul_mat_vec_a8`, the path these
types took before; Q4_K the same shape on the grouped kernel):

| shape n_out × K | Q3_K | Q5_K | Q6_K | Q8_0 | Q4_K |
|---|---|---|---|---|---|
| 1,024 × 5,120 | 6.7 µs, 335 GB/s | 8.9, 403 | 9.4, 458 | 11.2, 498 | 7.9, 373 |
| 6,144 × 5,120 | 24.7, 546 | 36.7, 589 | 42.2, 612 | 52.1, 642 | 30.7, 577 |
| 17,408 × 5,120 | 61.6, 622 | 94.8, 646 | 116.3, 629 | 148.3, 639 | 78.1, 642 |
| 5,120 × 6,144 | 25.3, 534 | 35.8, 603 | 41.5, 623 | 52.0, 643 | 30.1, 587 |
| 248,320 × 5,120 | 830.7, 658 | 1,288.0, 679 | 1,570.7, 664 | 1,961.6, 689 | 1,057.7, 676 |

| shape n_out × K, MMVQ | Q3_K | Q5_K | Q6_K | Q8_0 |
|---|---|---|---|---|
| 1,024 × 5,120 | 34.8 µs, 65 GB/s | 22.7, 159 | 37.9, 114 | 30.8, 181 |
| 6,144 × 5,120 | 148.1, 91 | 100.7, 215 | 155.1, 166 | 117.4, 285 |
| 17,408 × 5,120 | 420.0, 91 | 217.3, 282 | 392.6, 186 | 292.5, 324 |
| 5,120 × 6,144 | 147.8, 91 | 96.0, 225 | 141.4, 183 | 109.2, 306 |
| 248,320 × 5,120 | 6,096.6, 90 | 4,153.2, 210 | 5,677.2, 184 | 3,962.6, 341 |

## The wide forms: the chunked prefill's GEMM on the tiles

Above 32 rows a layer's run of tile-type shards goes through the grouped kernel's
wide forms — to 128 rows in place of the 32-row blocks, above them in place of the
dequantised tiles and cuBLAS bf16 (`gluon/interface.py`, the row-count dispatch):
one launch at split 1 (its descriptor table prepared at load beside the 1–32-row
splits), the grid over the tiles × M-blocks of 128 rows (one M-block to 128 rows,
ceil(M / 128) above it — sixteen for the 2,048-row chunk), eight warps (the mma's
warps [2, 4], the decode layouts replicated over the M pair, so every warp decodes
its 16 columns once and feeds them to 64 rows of int8
mma), the activations quantised once per 256 as in the decode (int8, fp32 scale,
int32 sums per 32), the A stages 2 × 128 × 256 bytes with 16-byte `cp.async` and
`ldmatrix` reads, the activation scale prefetched a k-block ahead. Region 9: 96 KB,
the A stages as the first allocation (Triton's allocator places the largest buffer
at offset 0) and the weight stages raw-addressed at word 16,384 of the second. Region
8 is the 64-row form on four warps (64 KB), taken from 33 to 64 rows. A run with
a Q4_K, Q5_K or IQ2_S shard keeps the 32-row blocks to 128 rows and the dequant
path above (`GLUON_DEQUANT_TYPES`): Q4_K and IQ2_S run at 0.7 × cuBLAS bf16 in
the wide form, so dequantising and calling cuBLAS is 1.2 × faster for them; Q5_K's
44-word rows take the wide stages of regions 5–7 and do not fit the wide forms.

The decode launches are unchanged. Their M-block offset is a plain zero and
folds away; the scale prefetch and the 16-byte swizzle are wide-form branches,
because carrying them into the decode forms costs 29 registers (a resident block
at M = 1 and at M = 8) and 4–5 % of the launch. Compiled for sm_89, the decode
variants keep the registers of the 32-row form above, and their machine code is
that of the 32-row form except for the row offsets, now int32 rather than int64:
40 to 64 instructions fewer at bm 16 and bm 32, byte for byte at M = 1.

Registers per wide variant (sm_89, `cuobjdump -res-usage` on a CPU compile of
the launch; stack bytes in brackets):

| variant | NEW = 0 | Q3_K | Q6_K | Q3_K + Q6_K + Q8_0 |
|---|---|---|---|---|
| 64 rows: bm 64, four warps, region 8 | 255 (16) | | | 255 (88) |
| 128 rows: bm 128, eight warps, region 9 | 255 (24) | 255 (104) | 255 (32) | 255 (96) |

`test_gluon_wide_gpu.py` is the GPU check (the card free, the command in its
docstring): the ten types at 17,408 × 5,120, 1,024 × 5,120, 1,024 × 17,408 and
48 × 5,120 (`n_valid` 48), 33 / 64 / 65 / 128 / 129 / 2,048 rows, both wide forms,
against the 32-row form on the same packed layer: 368 launches, 871,388,192
elements bit-identical, deterministic, NaN-free; against an fp64 GEMM on
`gguf.quants.dequantize`'s weights with the kernel's int8 activations max relative
error 1.5e-6 (the plugin's CUDA fp32 dequantiser is 6.4e-4 off `gguf.quants` on
Q4_K, exact on the others); a mixed layer of IQ3_S, IQ4_XS and Q6_K shards the same.
It then runs the op on a layer the loader prepared: at 129 and 300 rows a run of
wide-form shards leaves the wide launch bit-identical to the 32-row form, a run
holding a Q4_K shard leaves the dequant path, and a Q5_K layer is refused by the
launcher instead of being launched into a region its rows do not fit.

Per launch at 2,048 rows (the chunk), `--time`: CUDA-graph replays of ten launches,
median of nine, the card at its 249.6 W power limit (SW power cap active) — ms,
TOPS, the ratio of cuBLAS bf16 to the wide form, in brackets the ratio with the
tiles' dequantisation and the activation quantisation counted:

| type | 17,408 × 5,120 | 10,240 × 5,120 | 5,120 × 17,408 | 6,144 × 5,120 |
|---|---|---|---|---|
| IQ4_XS | 2.46 ms, 148 TOPS, 1.42× (1.61×) | 1.40, 153, 1.45× (1.62×) | 2.40, 152, 1.44× (1.53×) | 0.86, 149, 1.46× (1.59×) |
| IQ3_S | 2.26 ms, 162 TOPS, 1.55× (1.70×) | 1.33, 161, 1.51× (1.65×) | 2.15, 170, 1.57× (1.62×) | 0.76, 170, 1.69× (1.81×) |
| IQ3_XXS | 2.34 ms, 156 TOPS, 1.49× (1.67×) | 1.33, 162, 1.51× (1.66×) | 2.30, 159, 1.48× (1.55×) | 0.81, 159, 1.56× (1.68×) |
| IQ2_XXS | 2.58 ms, 141 TOPS, 1.34× (1.48×) | 1.52, 141, 1.31× (1.43×) | 2.54, 144, 1.32× (1.38×) | 0.93, 139, 1.34× (1.44×) |
| Q8_0 | 3.25 ms, 112 TOPS, 1.07× (4.10×) | 1.39, 154, 1.43× (5.54×) | 3.25, 112, 1.04× (3.91×) | 0.73, 176, 1.74× (6.17×) |
| Q6_K | 3.70 ms, 99 TOPS, 0.94× (1.10×) | 1.91, 113, 1.04× (1.22×) | 3.68, 99, 0.92× (1.04×) | 1.16, 111, 1.08× (1.23×) |
| IQ2_XS | 3.60 ms, 101 TOPS, 0.96× (1.07×) | 2.13, 101, 0.93× (1.03×) | 3.53, 103, 0.95× (1.01×) | 1.30, 99, 0.96× (1.04×) |
| Q3_K | 3.89 ms, 94 TOPS, 0.89× (1.00×) | 2.23, 96, 0.89× (1.00×) | 3.69, 99, 0.91× (0.98×) | 1.36, 95, 0.92× (1.00×) |
| IQ2_S (dequant path) | 4.98 ms, 73 TOPS, 0.70× (0.78×) | 2.96, 73, 0.68× (0.75×) | 4.95, 74, 0.69× (0.74×) | 1.79, 72, 0.70× (0.76×) |
| Q4_K (dequant path) | 5.17 ms, 70 TOPS, 0.69× (0.78×) | 2.94, 73, 0.70× (0.79×) | 4.72, 77, 0.71× (0.79×) | 1.77, 73, 0.70× (0.78×) |

cuBLAS bf16 on the dequantised weights takes 3.47–3.55 / 1.99–2.05 / 3.35–3.44 /
1.24–1.28 ms (101–109 TFLOPS) and cuBLAS int8 on the same shapes 0.94–1.28 /
0.54–0.70 / 0.88–1.31 / 0.34 ms (279–417 TOPS); the tiles' dequantisation
0.38–0.64 / 0.22–0.36 / 0.39–0.63 / 0.12–0.20 ms, Q8_0's 10.01 / 5.86 / 10.00 /
3.41 (the un-repack and the CUDA dequantiser); the activation quantisation
0.022–0.030 ms, 0.18 at K 17,408. The head, Q6_K 248,320 × 5,120 at 2,048 rows:
56.5 ms, 92 TOPS, 0.87× (1.07×), against cuBLAS bf16 49.4 ms and its
dequantisation 10.9 ms.

Below the chunk the row block is what decides. One bm-128 launch of 64 rows idles
half its rows, so the dispatch takes bm 64 up to 64 rows and bm 128 above: a batch
to 128 rows is one M-block and one pass over the weights, where the 32-row blocks
read them ceil(M / 32) times, and above 128 rows it is ceil(M / 128) passes. At 64
rows — eight streams behind a 7-token drafter — that is one against two. The wide
launch also runs at split 1 where a 32-row block ran at its table split, so the
fp32 partial sum the split needed is gone: in the profile below the sum kernel
drops from 467 launches per step to 188. A C8 torch profile of the step before the
change: the grouped kernel is 73.4 % of the GPU time at 591 launches per step for
about 300 layer ops, against Marlin's 404 launches for about 384 linears on the
W4A16 arm.
Measuring this on one layer understates it, because a 17,408 × 5,120 tile is 38 MB
and fits the 64 MB L2, so its second pass comes from cache, where a step streams
the layers' weights once — 12.5 GB of the 13.1 GB file.

The same C8 workload profiled before and after the change, same card, same curve,
same profiler: the grouped kernel 3,001.6 → 2,346.2 ms of a 4.09 → 3.43 s GPU-busy
window (73.4 % → 68.5 %), its launches 31,337 → 21,810 over 53 → 56 steps
(591 → 389 per step), the split-K sums 48.96 ms over 24,737 launches → 22.03 ms over
10,521, the profiled rate 338.6 → 395.7 tok/s. `bench/run_benchmarks.sh single`
after the change, second run, e2e tok/s at T=default and T=0: C1 101.3 / 104.2,
C2 163.0 / 176.6, C4 274.5 / 301.8, C8 348.3 / 423.2. One stream is 8 rows and
takes the unchanged path: labd at 100k reads a 72.5 s cold prefill.

What it is worth in the server, on the ByteShape GPU-5 file (`CTX=int4`, the int8
DFlash2 drafter, `max_num_batched_tokens` 2,048, the same card and probe before
and after): a cold 100k prefill 90.4 s → 69.6 s, the cached follow-ups of the same
document 2.25–2.26 s → 1.81–1.89 s at 8 output tokens and 6.17–6.42 s → 5.64–5.67 s
at 512 (they recompute one chunk), the KV pool 606,515 → 610,029 tokens (the
dequantised weight buffer no longer appears in the activation profile). Decode is
unchanged: labd 100k copy 182.4 → 186.8 tok/s, 4k copy 245.4 → 255.1. Quality:
GSM8K over 200 questions 0.970 → 0.965, and a passcode hidden at 10 %, 50 % and
90 % depth of a 100k context is retrieved in all three cases. The file's types make
this the FFN's gain: IQ4_XS, IQ3_S and IQ3_XXS are 73.5 % of its bytes and all take
the wide form, against Q4_K 12.7 % and Q5_K 1.6 % on the dequant path.

The first prefill after a cold Triton cache compiles the wide variants the model's
layers need — twelve for this file — and pays about 45 s for it once; the server
logs one `tiles_grouped_kernel` JIT warning, and later prefills of the same shape
run in 1.8–2.1 s. The startup profile run does not reach the wide form.
