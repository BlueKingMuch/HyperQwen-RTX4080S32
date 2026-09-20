# The patch series, one line each

What every file in `patches/` (and `kvarn/`) is, where it came from, and what retires it. The Dockerfile applies
them in the order of `patches/series` onto the installed vLLM wheel; `verify.sh` checks each one is in place. Kinds:

- **backport**: a merged or open upstream change carried early. Retires when the pin carries it.
- **fix**: a defect in upstream or in this stack, fixable upstream. Retires when upstream takes it.
- **feature**: something upstream does not have. Stays until upstreamed as a feature.
- **local**: this hardware or environment (WSL2, sm80, a tuned build, env knobs). Stays.
- **own**: a fix to a feature this repo introduced. Rides with that feature.

Cut against: the pin the current hunks were generated on. Every file is exported from its commit on the fork
branch (`cpuchip/vllm`, v0.29.0 + one commit per row, in series order, subject `[qwen38] <topic>`): **`qwen38/0.29` @ `337efb79f`** for every row except four, and **`qwen38/0.29-hq` @ `4879f94f3`** for `spec-decode-attn` (`711f8ac83`), `speed-knobs-envs` (`323e89b2f`), `triton-spec-attn-fp8-kv` (`83c9a589c`) — re-cut for the #114 registration moves on a new branch so the original export point stays unrewritten — and `memory-profile-after-warmup` (`bf29fa109`), cut there after that branch had already diverged,
so the series applies to the 0.29.0 tree with exact context; the Dockerfile, `patches/check_vllm_series.sh`,
`kvarn/install.sh` and `verify.sh` apply and check with `--fuzz 0`, and a hunk whose context has moved fails the
build by name instead of landing by guess. Regenerate a file with `bash scripts/export-patch.sh <fork checkout>
<commit> patches/<topic>.patch`; do not edit the files by hand. A patch that reads an env knob registers it in
`envs.py` in its own hunk (so the knob is in the torch.compile cache key), and reads it through `vllm.envs`.

Every change to the installed vLLM tree is a patch file, applied with `patch -p1 --fuzz 0`. No installer rewrites
installed sources by anchor matching, string replacement or code generation: `--fuzz 0` verifies the context it
applies to, an anchor-based rewrite does not, and each one that skips it has to carry a substitute — sha256 pins of
every file it reads, a reversibility seal, and a repin cascade when anything upstream of it moves. Measured on this
repo: the apply loop checks 48 patches in 0.07 s; the pins, seals and gates standing in for it across three
generator steps cost 72 s per build, and a change to `envs.py` under that scheme moves five pin locations. A step's
position in its `series` file expresses ordering; `fp8/fp8-paged` is two ordinary patch files applied after
`patches/series` and after `kvarn/install.sh`, so a patch works at every position in this pipeline.

A patch is active by default. The safety net is not a flag, it is that the patch breaks nothing else while
active — the shape, dtype and capability guard in the code is what makes that true, and it is the guard, not an
env knob, that has to be right. `_fp8_full_causal` in `v1/attention/ops/triton_unified_attention.py` is 24 ANDed
terms: sm_89, fp8-per-tensor KV, fp8 q/k/v, bf16 out, 24 query heads, 4 KV heads, head_size 256, block_size 880
or 896, causal, no sliding window, no softcap, no alibi, no sinks. Those 23 terms already refuse every other
configuration; the env flag in front of them adds only the ability to ship the patch inactive. An env knob is for
a choice that is genuinely open at the same shape, or for diagnostics. Where the better value follows from a
quantity known at launch — context length, batch size, draft count — that is a rule, not a switch.

A knob that restates a number the configuration already carries is worse than useless, because the two sources
diverge silently. `VLLM_TRITON_FP8_MQ3D_QMAX` is the literal `8` in `fp8/env.sh:42` while
`VLLM_SPEC_DECODE_ATTN_QMAX` is `DRAFT_TOKENS + 1` in `single-user/start_qwen.sh:321` and `spec_decode_attn.py`
caps at `min(64, max(n, BLOCK_M // G)) = 10`. All three agree only at `DFLASH_TOKENS=7`. At the documented
`DFLASH_TOKENS=15` the verify block is 16, the MQ3D gate `1 < max_seqlen_q <= QMAX` reads `1 < 16 <= 8`, and the
fast path turns off with nothing logged. Derive such a number from its one source.

| patch | kind | what | upstream | cut against | retires when |
|---|---|---|---|---|---|
| dflash2-backport | backport, RETIRED | DFlash2 speculator on 0.27.1 | vllm #52816 (in 0.28.0) | 0.27.1 | done; kept for history, skipped by the Dockerfile |
| dflash2-lookup-drafting | feature | lookup-augmented drafting for DFlash2 (n-gram search over the context) | none | 0.29.0, regenerated (32 hunks) | upstreamed |
| dflash2-ngram-chains | feature | quantized candidate chains for the drafter; `propose` override | none | 0.29.0 (`dp_sync` signature) | upstreamed |
| dflash2-prewarm | fix | compile every DFlash2 rung at boot instead of at first request | none yet | 0.29.0, regenerated (CP args on the launch path) | upstream PR |
| dflash2-z-adaptive-emitted | fix | adaptive z counts emitted tokens, not sampling slots | none yet | 0.29.0 | upstream PR |
| dspark-draft-quant-config | fix | bf16 DSpark drafter beside a quantized target (callable `hf_overrides`) | none yet | 0.29.0 | upstream PR |
| hybrid-kv-groups-v2-cudagraph | fix | KV group sizing when the smallest bucket is the drafter's sliding-window layers | none yet | 0.29.0; graph-reserve hunk retired (vLLM profiles it) | upstream PR |
| hybrid-sw-block-promote | fix | promote a draft SW layer's block to a divisor of the primary block instead of padding its page | none yet (upstream pads) | 0.29.0 (pad check mirrors upstream's non-MLA rule) | upstream PR |
| int4-kv-per-token-head | feature | int4 per-token-head KV cache with the DFlash2 drafter | none | 0.29.0; padded-page view hunk retired (layout strides) | upstreamed |
| mamba-align-checkpoint-order | fix | keep reachable Mamba state snapshots alive until request end (fork #52) | vllm #45238 (not merged) | 0.29.0 | check against upstream #52789 (internal prefill checkpoints, in 0.29) at each pin |
| mamba-align-retire-null-gaps | backport | align mode retires Mamba state blocks across null gaps instead of stopping at the first one (fork #101) | vllm #55450 (merged 2026-09-11, not in 0.29.0) | 0.29.0 (two hunks re-anchored around `_num_checkpoint_blocks`) | the pin that carries #55450 |
| mamba-chunked-prefill-align | fix | state loss and NaN during chunked prefill on Mamba/GDN | none yet | 0.29.0 | upstream PR |
| marlin-int8-layer-select | local | env vars to pick which layers run W4A8 with the Marlin kernel | none | 0.29.0 | stays |
| marlin-int8-negative-scales | fix | Marlin W4A8 reads group scales as unsigned; AutoRound exports negative ones | none yet | 0.29.0 | upstream PR |
| marlin-repack-staged-sm80 | local | one grow-only staging buffer for the sm80 Marlin repack (fork #27) | none | 0.29.0 | stays |
| marlin-tune-table | local | wiring for a locally built tunable Marlin extension, off by default | none | 0.29.0 source | stays |
| offload-dflash-eagle-groups | fix | OffloadingConnector under dflash flagged every KV group as draft attention (fork #33) | none yet | 0.29.0 | upstream PR |
| offload-mtp-serve | backport | OffloadingConnector serves stored hits under MTP/EAGLE instead of vetoing the request; load boundary from the computed offset; finished-request store watermark clamped (fork #100) | vllm #52771, #52807 (merged, not in 0.29.0) | 0.29.0 (all seven hunks unchanged) | the pin that carries both |
| offload-wsl2-devptr | local | CPU offload tier device pointers on WSL2 | none | 0.29.0 | stays |
| qwen3_5-embed-quant | fix | pass `quant_config` to the token embedding (main model and MTP module) | none yet | 0.29.0 | upstream PR |
| qwen3_5-mtp-draft-vocab | feature | vocab-truncated draft head for MTP | none | 0.29.0 | upstreamed |
| sampler-small-topk-fast-softmax | feature | sort-free top-k/top-p for small k, multi-block row softmax | none | 0.29.0 | upstreamed or superseded |
| spec-decode-attn | feature | split-KV verify attention on FLASH_ATTN with query-row tiling | none | 0.29.0 | upstreamed |
| engine-completion-log | feature | one log line per completed engine step, so a stalled core is visible without scraping stats gaps | upstream PR (syv-ai #94/#110) | 0.29.0 | upstreamed |
| engine-stall-sentinel | feature | daemon thread warns once per episode when no step completes for `VLLM_ENGINE_STALL_SENTINEL_S` while requests are live | upstream PR (syv-ai #94/#110) | 0.29.0, re-cut for the port | upstreamed |
| topk-honour-flashinfer-sampler-switch | fix | `VLLM_USE_FLASHINFER_SAMPLER=0` also covers the drafter's candidate top-k, which `_flashinfer_topk()` did not gate | none yet (syv-ai #106 B1) | 0.29.0 | upstream takes it |
| memory-profile-after-warmup | fix | run `profile_run` once before the memory-profiling window, synchronize and empty the allocator cache, so a cold compile cache's scratch is not counted as transient peak and the KV cache the warm boot grants is not refused | none yet | 0.29.0 | upstream profiles after warmup |
| cudagraph-memory-from-allocator | fix | measure captured CUDA-graph memory by the allocator's reserved bytes and log the driver's free-memory delta beside it; under WSL2's driver that delta reads zero once the KV cache fills the budget and collapses by 5.44 GiB during a cold compile, which the graph estimate subtracted from the KV budget and refused the CTX=huge first boot | none yet | 0.29.0 | upstream measures by the allocator |
| triton-spec-attn-fp8-kv | feature | split-KV verify attention on the per-tensor fp8 KV cache (TRITON_ATTN, sm89+); registers `VLLM_SPEC_ATTN_DEBUG` | none | 0.29.0, re-cut for the port | upstreamed |
| spec-decode-int4-kv-mq3d | feature | multi-query 3D int4 verify path | none | 0.29.0 | rides with int4-kv-per-token-head |
| int4-noncausal-tile-bounds | fix | the INT4 tile loop took the causal bound unconditionally: `compute_tile_loop_bounds` defaults `USE_CAUSAL=True`, and int4_per_token_head.py's call stopped at `IS_3D`. Non-causal attention lost every KV position past `context_len + (BLOCK_M-1)//G + 1` — at BLOCK_M 16, G 4 that is 4, and the DFlash2 drafter (`is_causal` false) runs a verify block of 8. Causal output bit-identical at every query length measured. The feature patch is upstream's (`04edd2b`, in `upstream/main`); its 0.28 form wires causality into the mask helper and does not touch the tile bound, which became a shared helper in 0.29 | none yet | 0.29.0 | upstream takes it |
| spec-decode-int8-kv | feature | split-KV verify attention over an int8 per-token-head cache | none | 0.29.0 | rides with spec-decode-attn |
| spec-decode-scratch-token-units | own | mq3d scratch sized in tokens, not sequences (fork #46, #57) | none | 0.29.0 | rides with mq3d |
| spec-decode-scratch-within-budget | own | mq3d scratch allocated inside the memory budget (fork #57) | none | 0.29.0 | rides with mq3d |
| spec-sampler-prewarm | fix | compile the rejection sampler's Triton kernels at boot (fork #48) | none yet | 0.29.0 | upstream PR |
| speed-knobs-envs | local | register this repo's env knobs in `envs.py` | none | 0.29.0 | stays while the knobs exist |
| prefill-attn-int8 | feature | int8-QK Triton prefill attention for head_dim 256 | none | 0.29.0 | upstreamed |
| vision-tower-cpu-offload | local | Qwen3 vision tower bulk weights in host RAM | none | 0.29.0 | stays |
| vllm-pr50021-gdn-spec-bounds | backport | bounds checks in GDN/KDA spec-decode state lookups | vllm #50021 (open) | 0.29.0 | the pin that carries #50021 |
| kvarn/kvarn-0.29.0 | feature | KVarN cache dtypes, quant mode, backend registration, page size | none (KVarN is Huawei CSL's, Apache-2.0) | 0.29.0; attn_utils view hunk retired | upstreamed |
| kvarn/kvarn-v2-runner-0.29.0 | own | KVarN with the V2 runner and DFlash2 (SW groups, Mamba block index, selector guards) | none | 0.29.0; kv_cache_utils hunks retired | rides with KVarN |

Below the `# --- Ada additions ---` header in `patches/series`, applied after everything above and exported from
`qwen38/0.29-on-148`, a branch rooted at the tree the series above leaves behind:

| patch | kind | what | upstream | cut against | retires when |
|---|---|---|---|---|---|
| gdn-spec-state-recovery-core | fix | accepted-token source indices reach the GDN and causal-conv1d kernels, so a speculative step recovers the recurrent state from the row it was written to | none yet | 0.29.0 + the series above | upstream PR |
| mamba-align-row-null-bounds | fix | accepted-token block-table columns masked against the request's own row width | none yet | 0.29.0 + the series above | upstream PR |
| gdn-persistent-recovery-buffer | own | the source-index buffer allocated once on the builder, so it survives graph capture | none | 0.29.0 + the series above | rides with gdn-spec-state-recovery-core |
| gdn-persistent-recovery-copy | own | that buffer filled during metadata build | none | 0.29.0 + the series above | rides with gdn-spec-state-recovery-core |
| gdn-active-runtime-k-width | own | spec masks sliced to the width the step uses | none | 0.29.0 + the series above | rides with gdn-spec-state-recovery-core |
| triton-fp8-mq3d-qmax8 | feature | opt-in FP8 multi-query 3D Split-KV; registers `VLLM_TRITON_FP8_MQ3D`, `_QMAX` | none | 0.29.0 + the series above | upstreamed |
| triton-fp8-mq3d-dispatch-trace | feature | opt-in INFO-once 2D/3D dispatch screening | none | 0.29.0 + the series above | rides with triton-fp8-mq3d-qmax8 |
| mq3d-mixed-target | feature | the FP8 MQ3D path on the target's attention only; registers `VLLM_TRITON_FP8_MQ3D_MIXED_TARGET` | none | 0.29.0 + the series above | rides with triton-fp8-mq3d-qmax8 |
| block-verification-invalid-draft | fix | NaN/+inf proposals survive the reductions as invalid and the block is discarded | none yet | 0.29.0 + the series above | upstream PR |

The two multi-query 3D paths in this tree do not compete. Upstream's is int4, under `VLLM_INT4_MQ_3D` in
`int4_per_token_head.py`; this one is fp8 per-tensor, under `VLLM_TRITON_FP8_MQ3D` in `triton_unified_attention.py`.
Disjoint knobs, disjoint files, and mutually exclusive by KV cache dtype. Where both gate the same call site --
`TritonAttentionImpl.forward()` -- upstream's `_spec_attn_run_fp8` is checked first and returns; the fp8 MQ3D gate
below it runs when `VLLM_SPEC_DECODE_ATTN` is off, which is its default.

Eight more from the same branch are **not** carried, because the series above already does the same thing:
`dflash-compressed-qkv-context-buffer` (the same WNA16 KV unpack in `qwen3_dflash.py`), `hybrid-kv-group-sizing` and
`hybrid-kv-group-capacity-cost` (`hybrid-sw-block-promote`), `dflash2-request-topk-topp`
(`dflash2-lookup-drafting` carries `req_top_p`/`req_top_k` in the kernel), `dflash2-prewarm` (in the series already),
`triton-zero-length-segment-guard` (`spec-decode-int4-kv-mq3d` adds byte-identical code to `reduce_segments()`),
`mamba-resume-block-size` (upstream resumes from `_mamba_spec.block_size` with a fallback instead of an assert) and
`vision-tower-cpu-offload` (both the knob and the `UVAOffloader` call in `qwen3_vl.py`).

## Not patch files

Three directories install after `patches/series` without going through it, and the Dockerfile runs them in this order.
`kvarn/` and `fp8/` change the vLLM package itself, `fp8/` after `kvarn/` because it pins bytes `kvarn/` leaves behind;
`gguf-plugin/` installs a separate package, touches nothing under `vllm/` and therefore pins nothing of this tree, which
is why it goes last:

| step | kind | what | how it is checked |
|---|---|---|---|
| `kvarn/install.sh` | feature | the KVarN KV cache, two patches plus its modules | two rows above; `verify.sh` reverse-dry-runs both patches |
| `fp8/install.sh` | feature | four FP8 Triton attention steps, all off by default (`fp8/README.md`) | each step pins what it reads and proves its output reverses to its parent; `verify.sh` re-derives the published seal from the archived parents |
| `gguf-plugin/install.sh` | feature | the out-of-tree GGUF plugin at a pinned commit, ten patches, twenty-six Gluon decode kernels | three hash gates (`gguf-plugin/README.md`); `verify.sh` checks the package, its extension and its model test |

`fp8/` is where the FP8 attention flags come from that `patches/triton-fp8-mq3d-qmax8.patch` and
`patches/mq3d-mixed-target.patch` above only half describe: those two register `VLLM_TRITON_FP8_MQ3D` and
`_MIXED_TARGET`, and `fp8/install.sh` adds `VLLM_TRITON_FP8_CAUSAL_FULL`, `_PREFILL_FLAT`, `_MQ3D_SEGMENTS` and
`_V_CHUNKED` beside them. `fp8/env.sh` turns the set on together; `CTX=fp8` and `KV=fp8triton` are the launcher
routes.

## Upstream fixes

Open upstream and carried here. Last in `patches/series`, cut against the tree everything above it leaves behind.

| patch | kind | what | upstream | cut against | retires when |
|---|---|---|---|---|---|
| renderer-clamp-max-tokens | fix | `VLLM_RENDERER_CLAMP_MAX_TOKENS`: a prompt that fits the context window is servable even when `prompt + max_tokens` would exceed it | vllm [#42474](https://github.com/vllm-project/vllm/issues/42474), open | 0.29.0 + the series above | upstream takes it |

Upstream refuses such a request with HTTP 400, because `max_input_tokens` is `max_total_tokens - max_output_tokens` and the prompt is checked against that. A client that sends a fixed safety cap therefore cannot use long contexts at all. The flag moves the check to the whole window in `_token_len_check`, and stops `get_encode_kwargs` capping tokenization below it so the real prompt length is visible to that comparison. Generation is unaffected either way: the engine already bounds it at the window. A prompt that genuinely does not fit is still rejected, and tokenization still stops one token past the window rather than reading all of an oversized prompt.

Default off. It is in `compile_factors`' ignore list because it shapes no generated code -- without that entry, toggling a validation-only switch would invalidate every compiled artifact, since `compile_factors()` starts from every env var and subtracts the ignored ones.

Retired at 0.29.0 and removed from the tree: `vllm-pr54282-draft-gumbel-salt` (vllm #54282, in 0.29.0),
`xgrammar-spec-terminated` (in 0.29.0), and `sse-keep-alive` (vllm 585bb07c7, in 0.29.0 and not in
0.28.0; the `--sse-keep-alive-interval` flag is unchanged, so nothing that sets it needs to change).

Retired on 0.29 for a different reason, and temporarily: `int4-mq3d-envs`: its two registrations
(`VLLM_INT4_MQ_3D`, `VLLM_INT4_MQ_3D_DEBUG`) already exist on this line in `speed-knobs-envs`, and
this line's readers already go through `vllm.envs`, so applying it duplicates them and fails at
`--fuzz 0`. The #114 restructure recreates it as its own topic, moving those registrations out of
`speed-knobs-envs` rather than adding a second copy, before the pin-flip PR. Until then the 0.28
and 0.29 shapes differ here by design.

Two files still carry raw `diff -ruN` headers with timestamps instead of a preamble (`dflash2-z-adaptive-emitted`,
`offload-wsl2-devptr`); their descriptions live in `docs/gotchas.md` and `docs/MR-DRAFT.md` until they get one.
