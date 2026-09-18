# A vLLM 0.29.0 patch series

Eighteen patches that apply to a pristine `v0.29.0` tree with exact context.
They are **not** the 0.28.0 series in `../` ported forward; they are a separate
series that happens to overlap it in two topic names. Nothing in this directory
is wired into the build: the Dockerfile, `verify.sh` and `patches/series` still
apply the 0.28.0 series against the `vllm==0.28.0` pin in
`docker/requirements.txt`, and this directory changes none of that.

It is here because [syv-ai/HyperQwen#106](https://github.com/syv-ai/HyperQwen/issues/106)
is open, and because a series that has been checked against the tag it claims is
worth more to that issue than an opinion about it.

[← back to the main README](../../README.md) · [the 0.28.0 series](../../PATCHES.md)

## Checking it

```
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git /tmp/vllm
bash patches/vllm-0.29/check_series.sh /tmp/vllm/vllm
sha256sum -c SHA256SUMS      # from inside this directory
```

Two passes, GNU `patch` then `git apply`, both at `--fuzz 0`. The 0.28.0 checker
can only afford its second pass on five patches; here all eighteen pass it,
because all eighteen are exported from commits rooted at the tag rather than
carried forward by hand. As measured: 18/18, zero hunks at an offset, zero at
fuzz, `git diff --check` clean, 19 files touched, +1028/-80.

## Provenance

Each file is exported from one commit on a branch of vLLM rooted at `v0.29.0`,
one commit per row below, subject `[qwen38] <topic>`, using the repository's own
`scripts/export-patch.sh`. The commit is the source of truth and the file is
generated from it:

```
FORK_NAME=qwen38/0.29 bash scripts/export-patch.sh <fork checkout> <commit> \
    patches/vllm-0.29/<topic>.patch
```

**Do not edit a file here by hand.** `SHA256SUMS` pins the reviewed bytes, and
`.gitattributes` marks `*.patch` as `-text` so a CRLF checkout on Windows cannot
change a hash. Fix the commit and re-export instead.

## The series

`kind` uses the vocabulary of [PATCHES.md](../../PATCHES.md). Its `retires when`
column is deliberately absent: for these patches that is not established, and a
guess in that column would read like a commitment.

| patch | kind | what | upstream |
|---|---|---|---|
| dflash-compressed-qkv-context-buffer | fix | DFlash context-KV fusion reads the compressed-tensors WNA16 loader representation; asymmetric and act-order layouts fail closed | none stated |
| vllm-pr50021-gdn-spec-bounds | backport | bounds checks in the GDN/Mamba spec-decode state lookups; kernel hunks only | vllm #50021 (open) |
| gdn-spec-state-recovery-core | fix | accepted-token source indices reach the GDN and causal-conv1d kernels, so a speculative step recovers state from the row it was written to | none stated |
| mamba-align-row-null-bounds | fix | accepted-token-derived block-table columns masked against the request's own row width | none stated |
| vision-tower-cpu-offload | local | Qwen3 vision-tower bulk weights in host RAM; registers `VLLM_VISION_CPU_OFFLOAD_GB` | none |
| gdn-persistent-recovery-buffer | own | the GDN source-index buffer allocated once on the builder, so it survives CUDA-graph capture | rides with gdn-spec-state-recovery-core |
| gdn-persistent-recovery-copy | own | that buffer filled during metadata construction, and passed on only when the step carries indices | rides with the buffer |
| hybrid-kv-group-sizing | fix | a padded sliding-window bucket preferred over splitting one, so the smallest bucket does not set the page size for its group | none stated |
| hybrid-kv-group-capacity-cost | fix | hybrid KV group layouts compared by allocator capacity cost, with `merge()` run on each real strided subset | none stated |
| dflash2-request-topk-topp | feature | per-request top-k/top-p for the DFlash speculators under CUDA graphs, staged into capture-stable buffers | none |
| gdn-active-runtime-k-width | fix | GDN spec masks sliced to the width the step uses, not the configured maximum | none stated |
| dflash2-prewarm | fix | every reachable `BLOCK_SIZE` variant compiled at capture instead of inside the first long request | none stated |
| triton-zero-length-segment-guard | fix | zero-length padding rows return a zero output row before the segment arithmetic, instead of entering `cdiv(0, 0)` | none stated |
| triton-fp8-mq3d-qmax8 | feature | opt-in FP8 multi-query 3D Split-KV, off by default; registers `VLLM_TRITON_FP8_MQ3D` and `VLLM_TRITON_FP8_MQ3D_QMAX` | none |
| triton-fp8-mq3d-dispatch-trace | feature | opt-in INFO-once 2D/3D dispatch screening; registers `VLLM_TRITON_FP8_MQ3D_TRACE`, excluded from compile factors | none |
| mq3d-mixed-target | feature | the FP8 MQ3D path on the target's attention only, draft unchanged; registers `VLLM_TRITON_FP8_MQ3D_MIXED_TARGET`, which does shape the graph | none |
| block-verification-invalid-draft | fix | NaN/+inf proposals survive the reductions as invalid, the block is discarded, and the target sampler is reused directly | none stated |
| mamba-resume-block-size | fix | Mamba state resumed in the Mamba group's own token geometry rather than the shared generic `block_size` | none stated |

Five knobs are registered in `envs.py` and read through `vllm.envs`, so they
enter the torch.compile cache key: `VLLM_VISION_CPU_OFFLOAD_GB`,
`VLLM_TRITON_FP8_MQ3D`, `VLLM_TRITON_FP8_MQ3D_QMAX`,
`VLLM_TRITON_FP8_MQ3D_TRACE` and `VLLM_TRITON_FP8_MQ3D_MIXED_TARGET`. Every
feature here is off by default.

## What is not here

The recipe this series comes from carries four more patches that this directory
does not, and the reason is structural rather than a matter of effort. They are
cut against a tree that three Python installers have already rewritten, and
those installers run between the patch steps. Without them the context those
four hunks expect does not exist, so they fail on a pristine `v0.29.0` tree both
individually and in series order. Carrying the four would mean carrying the
install steps, which are not part of this series.

No CPU test comes with the series either, and for the same reason. The source
recipe's exported tests are three, and all three are the reproductions those
four patches ship on -- the prefix-cache admission filter, the per-tile paged
addressing and the chunk-major V layout. None of them exercises anything in
these eighteen. Carrying them would mean shipping tests for code this fork does
not contain, which fails loudly at best and passes vacuously at worst.

So the CPU-side evidence here is the two apply passes and `SHA256SUMS`, not a
test suite. There is no GPU measurement behind any of this yet. The series
applies to the tag it names and the hashes match; that is the whole claim.

## What the jump costs the 0.28.0 series

Measured, not estimated, with `patch --fuzz 0` against shallow clones of both
tags. The 0.28.0 series minus its retired backport is 38 patches:

| series | against v0.28.0 | against v0.29.0 |
|---|---|---|
| the 0.28.0 series (38) | **38/38** | 18/38 |
| this series (18) | 12/18 | **18/18** |

The control matters: 38/38 on its own pin means the 20 failures on 0.29.0 are
the version jump and not patches that were already stale.

Three of those 20 are expected rather than work. [PATCHES.md](../../PATCHES.md)
already lists `vllm-pr54282-draft-gumbel-salt` and `xgrammar-spec-terminated` as
retiring at 0.29.0, and records that the CUDA-graph reserve hunk of
`hybrid-kv-groups-v2-cudagraph` retires when vLLM profiles V2 graphs, "0.29.0
does".

The other seventeen, in series order:

`spec-decode-attn`, `speed-knobs-envs`, `dflash2-lookup-drafting`,
`spec-decode-int8-kv`, `vision-tower-cpu-offload`, `int4-kv-per-token-head`,
`marlin-repack-staged-sm80`, `dflash2-ngram-chains`, `dflash2-prewarm`,
`prefill-attn-int8`, `spec-sampler-prewarm`, `mamba-align-checkpoint-order`,
`dflash2-z-adaptive-emitted`, `mamba-align-retire-null-gaps`, `sse-keep-alive`,
`int4-mq3d-envs`, `triton-spec-attn-fp8-kv`.

**Read that list as a count of patches that do not land, not as a count of
independent rebases.** The run is cumulative in `patches/series` order and
continues past a failure, so a patch whose context an earlier failed patch would
have added fails here on that account rather than on its own. The true number of
distinct rebases is somewhere below seventeen, and only doing them says where.

Two topic names appear in both series --- `dflash2-prewarm` and
`vllm-pr50021-gdn-spec-bounds`. They are different files cut against different
tags, and the 0.28.0 ones are the ones the build uses.
