# HyperQwen on an RTX 4080 SUPER (32 GB)

This is my experimental fork of [`syv-ai/HyperQwen`](https://github.com/syv-ai/HyperQwen),
focused on running and profiling it on a 32 GB RTX 4080 SUPER.

Most of the actual work is upstream. This repo contains the changes, experiments, and
measurements that came out of running it on my setup. I'm keeping it public in case any of
it is useful to other Ada users.

This is very much an experimental fork, almost completely vibe-coded, and I may stop working
on it at any point. If you want a maintained project, a known-good installation path, or
general documentation, use upstream and start with its [README](https://github.com/syv-ai/HyperQwen#readme).

If something from this fork is useful upstream, feel free to take it.

## Hardware and workload

My setup:

* RTX 4080 SUPER 32 GB (sm89, Ada)
* 250 W power limit, undervolted
* Windows 11
* WSL2 + Docker

Upstream's published numbers are from an RTX 3090 (sm86), so one of the main reasons for
this fork was simply to see how the same setup behaves on Ada.

The first baseline is upstream's own harness at `c0c81bb`, second run, setup B:

**[field report #149](https://github.com/syv-ai/HyperQwen/issues/149)**

Single-stream performance is a few percent below the published 3090 reference. At four and
eight concurrent requests, the 4080 SUPER is roughly a third faster. GSM8K scores 0.960.

While I tried to keep it comparably fast, single-stream decode isn't really the workload
I'm optimizing for, though.

This machine serves agentic workloads that fan out into multiple subagents: several
concurrent streams, long contexts, and mostly prefix-cached follow-up requests. That's the
workload the changes in this fork are aimed at.

## What's different

There are two main areas of work.

### GGUF and the Gluon kernels

The GGUF work originally started with the
[GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) checkpoint. Its
non-uniform quantizations are only available as GGUF, and the existing paths weren't fast
enough for the workload I wanted to run.

That led to the out-of-tree plugin and the Gluon kernels with eleven GGML quantization types
held in a tile-major layout and run by one grouped kernel, with the launch shape chosen by
the row count of the batch.

ByteShape's [GPU-5](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF) file came later. It
uses eleven quantization types at 3.84 bpw and is intended to stay closer to the unquantized
model. That's the checkpoint I'm currently using/testing.

Both files load and work, but their type mixes differ, which changes which path individual layer
shards take. Results from one checkpoint therefore shouldn't be assumed to apply to the
other, because I also had to learn this the hard way: just because a byte format is smaller
does not mean the GPU can work with it faster.

Drafters as well: `DRAFT=` now accepts a DFlash2 block drafter as GGUF, in llama.cpp's own
`dflash` architecture rather than a repack of the HF checkpoint. Tested on this machine
against the W4A16 drafter it would replace.

Layouts, register tables, per-shape measurements, profiles, and the patch series are in
[`gguf-plugin/README.md`](gguf-plugin/README.md).

The plugin source itself isn't vendored into this repository. `install.sh` downloads the
archive and verifies its SHA-256. It has no runtime effect unless `MODEL=` or `DRAFT=`
points to a `.gguf` file.

#### Results of tested drafters

Teacher-forced acceptance, `bench/labd_accept.py --chunk 32` at `DFLASH_TOKENS=7`, two runs
each, as accepted tokens per step. The ladder is Anbeeld's; the second Q4_K_M is z-lab's
separate conversion of the same drafter, and W4A16 is the drafter the launcher picks when
`DRAFT=` is empty.

| drafter | copy | code | edit | quote | summary | qa |
|---|---|---|---|---|---|---|
| [bf16](https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-bf16.gguf) | 7.93/7.85 | 3.64/3.64 | 1.91/1.91 | 4.32/4.19 | 2.09/2.09 | 2.53/2.53 |
| [Q8_0](https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q8_0.gguf) | 7.86/7.85 | 3.92/3.92 | 1.91/1.91 | 4.35/4.21 | 2.11/2.11 | 2.42/2.50 |
| [Q6_K](https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q6_K.gguf) | 7.93/7.94 | 4.00/4.00 | 1.68/1.50 | 4.26/4.25 | 2.06/2.06 | 2.45/2.48 |
| [Q5_K_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q5_K_M.gguf) | 7.85/7.83 | 3.64/3.64 | 1.91/1.91 | 4.16/4.34 | 2.11/2.11 | 2.41/2.51 |
| [Q4_K_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | 7.84/7.83 | 3.57/3.57 | 1.97/1.97 | 4.06/4.33 | 2.12/2.12 | 2.42/2.39 |
| [Q3_K_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q3_K_M.gguf) | 7.93/7.93 | 3.57/3.85 | 1.43/1.43 | 4.21/4.35 | 2.05/2.05 | 2.49/2.52 |
| [Q2_K](https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q2_K.gguf) | 7.84/7.85 | 2.67/2.67 | 1.91/1.91 | 4.47/4.52 | 1.96/1.96 | 2.30/2.39 |
| [Q4_K_M, z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | 7.92/7.93 | 3.85/3.85 | 1.97/1.97 | 4.22/4.26 | 2.10/2.10 | 2.32/2.24 |
| [W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16) | 7.93/7.92 | 3.57/3.60 | 1.43/1.43 | 4.24/4.24 | 2.14/2.14 | 2.58/2.55 |

I decided to stay on the W4A16 drafter, because it has the best ratio of VRAM footprint to
quality and speed.

### Ada-specific experiments

This fork also carries a few experiments specifically for my 4080 SUPER setup:

* vLLM 0.29.0 instead of 0.28 — upstream's own port
  ([syv-ai/HyperQwen#148](https://github.com/syv-ai/HyperQwen/pull/148)) is the base, and
  this fork adds nine patches on top of its 38 ([`PATCHES.md`](PATCHES.md))
* int4 per-token-head KV cache on the Triton backend via `CTX=int4`
* four FP8 Triton attention changes under [`fp8/`](fp8/README.md)

The attention changes and the int8 layer-select path are off unless something turns them on,
and `verify.sh` checks that behaviourally, against the live environment registry rather than
against the source text.

These changes are deliberately narrow.

They were tuned and measured on one GPU: an sm89 RTX 4080 SUPER with 32 GB VRAM, a 250 W
power limit, and an undervolt.

The attention changes only apply to this geometry:

* 24 query heads
* 4 KV heads
* head dimension 256
* block size 896

For other backends or dtypes, the original paths are used.

A 4090 shares the Ada architecture, but it has different memory capacity and bandwidth. 
I don't have one, so the numbers in this repository should not be treated as 4090 or other
cards' results.

## Three checkpoints, one envelope

Everything below was measured on this machine with the same serving configuration,
one checkpoint at a time. In config, only `MODEL=` differed between the runs, 
but following stayed the same:

* `CTX=int4` — int4 per-token-head KV cache on the Triton backend
* `SPEC=dflash2` with 7 draft tokens and an int8 per-token-head drafter cache
* `MAX_SEQS=8`, `GPU_UTIL=0.88`, `max_num_batched_tokens` 2048
* `max_model_len` 262,144, prefix caching on, `MAMBA_SSM_DTYPE=bfloat16`
* vision tower loaded

The full annotated configuration is [`.env.example`](.env.example).

### Speed

`bench/run_benchmarks.sh single`, the harness from upstream: one warm-up run, then
the run that counts. End-to-end tok/s.

| | GPU-5 | GSQ-RCO IQ3_S | W4A16 AutoRound-fast |
| --------------- | ---------------: | ------------: | -------------------: |
| C1 T=default    |           101.28 |        108.22 |               108.25 |
| C2 T=default    |           162.99 |        165.05 |               173.93 |
| C4 T=default    |           274.47 |        280.10 |               318.61 |
| C8 T=default    |           348.29 |        350.96 |               409.42 |
| C1 T=0          |           104.20 |        114.06 |               115.30 |
| C2 T=0          |           176.61 |        181.78 |               186.17 |
| C4 T=0          |           301.75 |        298.28 |               328.64 |
| C8 T=0          |           423.21 |        388.41 |               387.96 |
| cold 100k prefill |          72.5 s |        72.8 s |               80.0 s |
| KV pool at 262,144 |        610,029 |       636,735 |              574,889 |


### Quality

GSM8K is `bench/quality_battery.py --gsm-only`, n=200. IFBench is 300 prompts per
run, prompt-level, reported as the harness reports it: strict / loose.

| | GPU-5 | GSQ-RCO IQ3_S | W4A16 AutoRound-fast |
| ------------------ | ----------: | ----------: | ----------: |
| GSM8K              |       0.960 |       0.950 |       0.970 |
| IFBench, no thinking | 41.3 / 43.0 | 38.0 / 41.7 | 41.0 / 43.3 |
| IFBench, low       | 59.0 / 68.0 | 56.3 / 67.3 | 53.0 / 61.7 |
| IFBench, medium    | 59.3 / 67.0 | 56.3 / 66.3 | 57.7 / 66.3 |
| IFBench, xhigh     | 75.7 / 83.3 | 74.3 / 82.0 | 72.3 / 79.7 |

The gap between the strict and the loose column is small without thinking and large
with it. IFBench's loose scoring accepts a response with its first or last line
removed; a reasoning parser leaves the separator newlines in front of the answer,
and strict scoring counts those.

## Benchmarks and reproductions

Measured runs are under [`docs/reproductions/`](docs/reproductions/README.md).

Each run starts cold in its own project and includes the configuration used to produce it.

Additional notes and results:

* [`docs/long-context.md`](docs/long-context.md)
* [`docs/benchmarks.md`](docs/benchmarks.md)

I'm trying to keep measured results separate from assumptions here. If a number isn't backed
by a run in the repository, it shouldn't be treated as a benchmark result.

## Building

`single` and `batch` sit behind Compose profiles, so the one command that builds the image
and starts the server is:

`docker compose --profile single up -d`

`pull_policy: build` means it rebuilds whenever the build context changed and reuses the
layer cache when it did not. To build without starting anything, name the service:
`docker compose build single`.

The install gate runs inside the build, and against a built image with:

`bash verify.sh --install`

`docker-compose.yml` intentionally builds the image locally instead of pulling upstream's
published image. The upstream image uses a different stack despite having a similar-looking
name.

For installation, start with:

[`docs/install.md`](docs/install.md)

Things that broke badly enough to be worth documenting are in:

[`docs/gotchas.md`](docs/gotchas.md)

## License

Apache-2.0, inherited from upstream.

The Qwen3.8-27B weights and the quantized checkpoints used here are also published under
Apache-2.0. The patches included in this repository are derivative works of vLLM, which is
Apache-2.0 as well.
