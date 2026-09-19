# The FP8 Triton attention steps

Four steps that rewrite vLLM's Triton attention for one geometry: 24 query
heads, 4 KV heads, head_dim 256, KV block 880 or 896, fp8 per-tensor KV, sm_89.
That is this model on this card. On any other backend, dtype or shape the gates
below are false and the original loops run unchanged.

The first three are **not** patch files. Each rewrites installed sources by
exact anchor matching, guards itself with sha256 pins of every file it reads,
and proves its output reverses byte-for-byte back to its parent before writing
anything. The fourth, `fp8-paged`, is two ordinary patch files with neither
property; what stands in for the reverse proof there is `fp8-paged/PINS`, which
hashes all four target files before and after each patch. A pin that does not
match means the tree under it moved; that is the signal, not an inconvenience.

[← back to the main README](../README.md) · [the vLLM series](../PATCHES.md)

## Running it

```
bash fp8/install.sh --check    # verify what is carried here; installs nothing
bash fp8/install.sh            # gate, install, gate again
```

The Dockerfile runs it after `patches/series` and after `kvarn/install.sh`,
because all four steps pin bytes of files those two touch. The order of the
steps themselves is in `series` and is not a preference: each reads what the one
before it wrote.

## The four steps and their flags

Every flag is off by default, and `verify.sh` checks that in the live registry
rather than in the source. The check is registration: the names exist and read
as off. The CPU gates prove the generated source reverses to its parent.

| step | flag | what it changes |
|---|---|---|
| `fp8-causal` | `VLLM_TRITON_FP8_CAUSAL_FULL` | full-causal FP8 attention, 2D and 3D routes |
| `fp8-causal-r2` | the same flag | narrows it to 2D; every 3D route keeps the original loop |
| `fp8-composite` | `VLLM_TRITON_FP8_PREFILL_FLAT` | flat prefill index mapping |
| | `VLLM_TRITON_FP8_MQ3D_SEGMENTS` | 32 parallel softmax segments instead of 16 |
| `fp8-paged` | `VLLM_TRITON_FP8_V_CHUNKED` | V stored as one 32-token tile per chunk inside a KV block |

Two more flags come from the vLLM series rather than from here, and the
launchers set them with the rest: `VLLM_TRITON_FP8_MQ3D` and
`VLLM_TRITON_FP8_MQ3D_MIXED_TARGET` (`patches/triton-fp8-mq3d-qmax8.patch`,
`patches/mq3d-mixed-target.patch`).

`fp8/env.sh` turns the set on together, because that is how it was measured: the
flat mapping needs full-causal, the 32-segment selector needs MQ3D, and
V-chunked needs the 896 block size the launcher passes with it.

```
CTX=fp8 bash single-user/start_qwen.sh      # single-user
KV=fp8triton bash batch/start_qwen.sh       # batch
```

Setting any of the seven variables yourself keeps your value: `fp8/env.sh` only
fills in what is unset.

## What is not here

`VLLM_MAMBA_ALIGN_SPARSE_RECLAIM`, a third feature the seal could have carried.
This tree already retires those blocks:
`patches/mamba-align-retire-null-gaps.patch` overrides
`_remove_blocks_in_range` on the Mamba manager subclass, delegating to `super()`
outside `mamba_cache_mode == "align"`, with no flag to turn it on. A second
definition would shadow it.
`install_composite.py` says so in its manifest, under `not_installed_here`, and
the CPU gate asserts both halves of that claim against the tree.

## The scratch pool

vLLM 0.29 moved the three 3D-softmax scratch buffers into a pool that
`mq3d_scratch_plan()` sizes, and that plan sized rows with the module constant
`NUM_PAR_SOFTMAX_SEGMENTS = 16`. A builder selecting 32 segments against a pool
built for 16 writes past the end of all three buffers.

The pool key already carries a slot for the segment count — upstream's own
comment over `_MQ3D_SCRATCH_POOL` names it — so `install_decode_segments.py`
makes the count a field of the plan and every user reads the plan instead of the
constant. At the default the selected count *is* the constant, so the pool, its
key and its byte count are what they were. With the flag on, the attention Impl
still plans with the default inside the memory profile, the builder's key
therefore differs, the builder allocates its own set after the profile, and
upstream's own WARNING prints the byte count. That is the cost of 32 segments,
in the boot record, where it can be read back.

## Pins

| gate | what it says |
|---|---|
| `SHA256SUMS` | what this repo carries is what was reviewed |
| each step's `PINS` / `PARENT_SHA` | the files it read entered in the exact state it was cut for |
| each step's reverse proof | its output differs from its parent in exactly the replacements it declares |
| `fp8-paged/PINS` | the two patch files' four targets, before and after each patch |

To move the whole set onto a different tree:

```
python scripts/repin-fp8-installers.py fp8 <vllm package dir>
bash fp8/install.sh --write-sums
FP8_PINS=write bash fp8/install.sh
bash fp8/install.sh --write-sums
```

The first prints every pin that moved, in dependency order. A pin that moves is
a file something upstream of it changed — look at why before trusting the build.
The third re-runs the series and rewrites `fp8-paged/PINS` from the tree.

`--write-sums` appears twice because every pin in this directory is itself
content-hashed by `SHA256SUMS`, and that gate is the first thing `install.sh`
checks. Rewriting a pin — by repinning, or by `FP8_PINS=write` rewriting
`fp8-paged/PINS` — leaves it stale and the next run refuses. `--write-sums` is
the one mode that does not check the gate first.
