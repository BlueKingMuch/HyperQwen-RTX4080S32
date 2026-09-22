# HyperQwen on an RTX 4080 SUPER (32 GB)

A personal experiment fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen),
kept public in case it is useful to someone else with an Ada card. 
Everything that makes this work is upstream. 
What happens here is one card, one setup, and what I can measure on it.

**This is a vibed fork and may be abandoned at any time.** 
For a maintained project, a working install and the documentation, go upstream. 
Start with its [README](https://github.com/syv-ai/HyperQwen#readme). 
If anything here turns out to be worth keeping, the maintainer is welcome to take it; that is what the fork
link is for.

## The card

RTX 4080 SUPER 32 GB (sm89, Ada), capped at 250 W and undervolted, Windows 11 with WSL2 and Docker. 

Upstream's published numbers are an RTX 3090 (sm86), so the question this fork exists to answer is what transfers and what does not.

First data point, upstream's own harness at `c0c81bb`, second run, setup B:
[field report #149](https://github.com/syv-ai/HyperQwen/issues/149).

Single stream lands a few percent below the 3090 reference because decode is bandwidth-bound there and the 3090 has more of it.
Four and eight concurrent requests run roughly a third faster. GSM8K 0.960.

## What is different here

Three things, all in the build, all off unless something turns them on.

**vLLM 0.29.0 instead of 0.28.** Upstream's own port
([syv-ai/HyperQwen#148](https://github.com/syv-ai/HyperQwen/pull/148)) is the
base; `patches/series` carries its 38 entries and then nine more under an
`# --- Ada additions ---` header. `patches/check_vllm_series.sh` takes all 47
against a pristine `v0.29.0` and reports 46 applied with exact context, 5 of them
at an offset, 0 with fuzz. Eight further patches from the same branch are not
carried because upstream's series already does the same thing;
[PATCHES.md](PATCHES.md) names each one.

**`fp8/` — four FP8 Triton attention steps.** `CTX=fp8` (single-user) or
`KV=fp8triton` (batch) serves the fp8 KV cache on the Triton backend with
full-causal attention, a flat prefill index mapping, 32 parallel softmax
segments instead of 16, and V stored as one 32-token tile per chunk inside a KV
block. All four default off and `verify.sh` checks that in the live environment
registry. They apply to one geometry — 24 query heads, 4 KV heads, head_dim 256,
block 896, sm89 — and on any other backend or dtype the original loops run.
What the dtype buys on this card is the pool: 308,331 KV tokens against bf16's
68,605 at the same 64k window. [fp8/README.md](fp8/README.md) has the pins, the
reverse-byte proofs, and the one place where vLLM 0.29's scratch pool had to
learn a second segment count.

**`CTX=int4` — the int4 per-token-head KV cache on the Triton backend.** Two
patches under the same header: the verify kernel feeds Q and K to the tensor
cores as int8 and P and V as bf16 instead of fp32 on TF32, in one query block
per request (kernel per launch at 100k: 1.964 ms to 0.690, against the bf16
split-KV kernel's 0.714; 0.187 to 0.0082 rel. RMS against an fp32 reference),
and prefill chunks dequantize the request's cached K/V per layer and run
FlashAttention-2 on it (first 100k prefill 165 s to 79, bf16 77). The DFlash2
drafter keeps an int8 per-token-head cache, whose page divides the int4 page
at block 1696. On this card the arm decodes 100k context at 39.9 ms per step
against bf16's 39.5 with 574,889 KV tokens at a 262k window; GSM8K over 200
questions 0.945-0.955 against bf16's 0.965.

**`gguf-plugin/` — the out-of-tree GGUF plugin, installed by default.** Pinned
commit, ten patches, twenty-six Gluon decode kernels (eleven tile types on the
grouped one, whose wide form runs the prefill chunks' GEMMs on int8 tensor cores
from the tiles instead of dequantising them: a cold 100k prefill of the ByteShape
GPU-5 file falls from 90.4 s to 69.6 s), its CUDA extension built for sm_89 and
sm_120 at image build. None of the plugin's source is carried
here; `install.sh` fetches the archive and checks its sha256. It costs nothing at
run time until a `MODEL=` path ends in `.gguf`, which is the whole of the
plugin's claim on a model. The weights are not produced by this repository.

`docker compose build` builds all of it and `verify.sh --install` is the gate.
`docker-compose.yml` builds locally rather than pulling upstream's published
image, because that image is a different stack under a name that looks like this
one.

## What it measures

[docs/reproductions/ada-029-fp8-gguf.md](docs/reproductions/ada-029-fp8-gguf.md)
has six arms, each started cold in its own project, with the numbers and the
configuration that produced them.

The short version, on this card:

- **`CTX=fast` is unchanged** by any of the above. The additions cost it nothing.
- **`CTX=fp8`** reaches 504 tok/s decode at eight concurrent requests against
  396, with mean TTFT at 888 ms against 3,172, and a 4.5x KV pool. Its GSM8K in
  four runs was 0.915 / 0.955 / 0.960 / 0.965, where bf16 and int8 stayed inside
  0.950 to 0.965.
- **`CTX=long`**, upstream's arm and untouched here, reaches 736 tok/s decode at
  eight concurrent requests, the highest of any arm measured here.
- **The GSQ-RCO IQ3_S checkpoint**, served through the plugin, is 5.6% faster at
  one concurrent request and 39% slower at eight, with GSM8K indistinguishable
  from W4A16 and 62% more KV pool at the same memory quota. That comparison is
  one variable: same image, same envelope, different weights.

The token cap the quality harness uses truncates 5 to 9 of its 200 questions in
every arm measured here, including the baseline.

## Licence

Apache-2.0, inherited from upstream. The Qwen3.8-27B weights and the quantised
checkpoints used here publish under Apache-2.0 as well, and the patches carried
here are derivative works of vLLM, also Apache-2.0.
