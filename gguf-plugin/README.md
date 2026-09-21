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

`gguf-plugin/test_gluon_kq_gpu.py` runs on the card (its docstring has the `docker run`).

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
