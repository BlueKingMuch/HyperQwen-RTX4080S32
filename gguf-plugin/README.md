# The out-of-tree GGUF plugin, pinned and patched

The target's non-uniform GSQ-RCO quantizations exist only as GGUF, and vLLM
reads GGUF through an out-of-tree plugin. This directory is what it takes to get
that plugin to serve them: the plugin fetched at a pinned commit, ten patches,
twenty-six Gluon decode kernels, and a CPU gate that says the result installed.

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
