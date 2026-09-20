# The FP8 Triton attention steps

Four steps that rewrite vLLM's Triton attention for one geometry: 24 query
heads, 4 KV heads, head_dim 256, KV block 880 or 896, fp8 per-tensor KV, sm_89.
That is this model on this card. On any other backend, dtype or shape the gates
below are false and the original loops run unchanged.

All four are patch files, applied with `patch -p1 --fuzz 0`. A hunk whose
context has moved fails the build by name instead of landing by guess, so the
patch's own context is what says the tree underneath it is the one each step
was cut against.

[← back to the main README](../README.md) · [the vLLM series](../PATCHES.md)

## Running it

```
bash fp8/install.sh --check    # the patch files are present; installs nothing
bash fp8/install.sh            # apply the four steps
```

The Dockerfile runs it after `patches/series` and after `kvarn/install.sh`,
because all four steps touch files those two touch. The order of the
steps themselves is in `series` and is not a preference: each reads what the one
before it wrote.

## The four steps and their flags

Every flag defaults to the value that leaves the original path in place, and
`verify.sh` checks that in the live registry rather than in the source. The
check is registration: the names exist and hold their defaults. Four of the
seven are booleans defaulting `False`; `VLLM_TRITON_FP8_PREFILL_FLAT` is `0`,
`VLLM_TRITON_FP8_MQ3D_SEGMENTS` is `16` — upstream's own count — and
`VLLM_TRITON_FP8_MQ3D_QMAX` is `8`.

| step | flag | what it changes |
|---|---|---|
| `fp8-causal` | `VLLM_TRITON_FP8_CAUSAL_FULL` | full-causal FP8 attention, 2D and 3D routes |
| `fp8-causal-r2` | the same flag | narrows it to 2D; every 3D route keeps the original loop |
| `fp8-composite` | `VLLM_TRITON_FP8_PREFILL_FLAT` | flat prefill index mapping |
| | `VLLM_TRITON_FP8_MQ3D_SEGMENTS` | 32 parallel softmax segments instead of 16 |
| `fp8-paged` | `VLLM_TRITON_FP8_V_CHUNKED` | V stored as one 32-token tile per chunk inside a KV block |

Three more flags come from the vLLM series rather than from here, and the
launchers set them with the rest: `VLLM_TRITON_FP8_MQ3D`
(`patches/triton-fp8-mq3d-qmax8.patch`), `VLLM_TRITON_FP8_MQ3D_MIXED_TARGET` and
`VLLM_TRITON_FP8_MQ3D_QMAX` (both `patches/mq3d-mixed-target.patch`). Four here
plus three there is the seven `fp8/env.sh` fills in.

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

`VLLM_MAMBA_ALIGN_SPARSE_RECLAIM`, a third feature these steps could have carried.
This tree already retires those blocks:
`patches/mamba-align-retire-null-gaps.patch` overrides
`_remove_blocks_in_range` on the Mamba manager subclass, delegating to `super()`
outside `mamba_cache_mode == "align"`, with no flag to turn it on. A second
definition would shadow it.
`fp8-composite` therefore leaves that function alone; nothing in this directory
defines it a second time.

## The scratch pool

vLLM 0.29 moved the three 3D-softmax scratch buffers into a pool that
`mq3d_scratch_plan()` sizes, and that plan sized rows with the module constant
`NUM_PAR_SOFTMAX_SEGMENTS = 16`. A builder selecting 32 segments against a pool
built for 16 writes past the end of all three buffers.

The pool key already carries a slot for the segment count — upstream's own
comment over `_MQ3D_SCRATCH_POOL` names it — so `fp8-composite` makes the count
a field of the plan and every user reads the plan instead of the constant. At the default the selected count *is* the constant, so the pool, its
key and its byte count are what they were. With the flag on, the attention Impl
still plans with the default inside the memory profile, the builder's key
therefore differs, the builder allocates its own set after the profile, and
upstream's own WARNING prints the byte count. That is the cost of 32 segments,
in the boot record, where it can be read back.

## Gates

| gate | what it says |
|---|---|
| `patch --fuzz 0` | every hunk's context is exactly the tree the step was cut against |
| the structural greps in `install.sh` | the named definitions are where the steps put them |
| the live `vllm.envs` registry | `install.sh` reads four flags back at their defaults, `verify.sh` all seven |

To move the whole set onto a different tree, re-cut each step from its commit on
the fork branch with `scripts/export-patch.sh`, the same way `patches/` is
regenerated. Patch files are never edited by hand.

Nothing here hashes a file that git already carries: `--fuzz 0` covers the tree
the patches land on, and git covers the patches themselves.
