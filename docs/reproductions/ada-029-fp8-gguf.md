# Reproduction: vLLM 0.29, the FP8 attention steps, and GSQ-RCO on an RTX 4080 SUPER

Same box and the same protocol as [wsl2-4080-super.md](wsl2-4080-super.md),
which measured vLLM 0.28. Six arms, each in its own compose project with its
own compile-cache volume, each started cold and torn down afterwards.

[← back to the reproductions index](README.md)

## Hardware and stack

| | |
|---|---|
| GPU | RTX 4080 SUPER, 32,760 MiB, 250 W cap, undervolted |
| OS | Windows 11 Pro 26200, WSL2, Docker |
| vLLM | 0.29.0, torch 2.13.0+cu130, Triton 3.7.1 |
| image | `docker compose build` from this branch; `verify.sh --install` passes with 0 failures |
| `.env` | `SPEC=dflash2`, `PREFIX_CACHE=1`, `VLLM_WSL2_ENABLE_PIN_MEMORY=1` |
| protocol | `bench/run_benchmarks.sh single` twice, second counts; `bench/quality_battery.py <tag> --gsm-only` four times |

## The arms

| tag | weights | KV | backend | window / seats |
|---|---|---|---|---|
| `finbase` | W4A16-AutoRound-fast | bf16 | FLASH_ATTN | 65,536 / 8 |
| `finlong` | W4A16-AutoRound-fast | int8 per-token-head | TRITON_ATTN | 131,072 / 4 |
| `shipfp8` | W4A16-AutoRound-fast | fp8_e4m3 | TRITON_ATTN | 65,536 / 8 |
| `parity-single` | GSQ-RCO IQ3_S (GGUF) | fp8_e4m3 | TRITON_ATTN | 65,536 / 8 |
| `parity-w4a16` | W4A16 | fp8_e4m3 | TRITON_ATTN | 65,536 / 8 |
| `029-upstream` | W4A16-AutoRound-fast | bf16 | FLASH_ATTN | 65,536 / 8 |

`029-upstream` is `main` after the 0.29 merge, before this branch's additions.
The two `parity-*` arms are the same envelope with different weights.

## KV pool

| arm | tokens | window |
|---|---|---|
| `finbase` bf16 | 68,605 | 65,536 |
| `finlong` int8 | 136,429 | 131,072 |
| `shipfp8` fp8 | 308,331 | 65,536 |
| `parity-single` fp8 + IQ3_S | 297,408 | 65,536 |
| `parity-w4a16` fp8 + W4A16 | 183,500 | 65,536 |

At one memory quota the 3-bit weights leave 62% more KV pool than W4A16
(297,408 against 183,500).

## Throughput

`bench/run_benchmarks.sh single`, second run. decode is `concurrency / mean TPOT`.

| cohort | `029-upstream` | `finbase` | `finlong` | `shipfp8` |
|---|---|---|---|---|
| C1 default | 106.7 | 114.4 | 119.5 | 116.8 |
| C2 default | 207.3 | 218.8 | 209.4 | 208.1 |
| C8 default | 451.7 | 396.2 | 736.0 | 504.4 |
| C1 greedy | 116.8 | 114.9 | 117.8 | 120.9 |
| C2 greedy | 218.6 | 213.2 | 224.5 | 216.7 |
| C8 greedy | 426.0 | 451.0 | 789.0 | 529.8 |

Mean TTFT at C8, default sampling: `finbase` 3,172 ms, `shipfp8` 888 ms.

## GSM8K

`bench/quality_battery.py --gsm-only`, n=200, four runs.

| arm | runs | span |
|---|---|---|
| `finbase` | 0.955 0.960 0.965 0.955 | 0.010 |
| `finlong` | 0.950 0.960 0.960 0.960 | 0.010 |
| `shipfp8` | 0.915 0.955 0.960 0.965 | 0.050 |
| `parity-single` | 0.960 0.955 0.945 0.955 | 0.015 |
| `parity-w4a16` | 0.955 0.965 0.955 0.950 | 0.015 |
| `029-upstream` | 0.965 | — |

The repository's published band across every shipped configuration is 0.950 to
0.965 at n=200, with a stated standard error of about 1.3 points
(`docs/gotchas.md`, 40) and a 1.0-point difference at n=200 called unresolvable
(`docs/spec-decode-scratch-token-units.md`). Greedy output is not bit-stable
here: `docs/gotchas.md` 13, 27, 48 and 50 record the mechanisms, and the battery
dispatches its 200 questions through `ThreadPoolExecutor(32)` with no seed, so
batch composition differs on every run.

## The 768-token cap

`bench/quality_battery.py` sends `max_tokens: 768` and never reads
`finish_reason`. When the `Final answer:` regex misses, the score falls back to
`extract_num`, which takes the last number anywhere in the text, so a truncated
answer scores on whatever figure it stopped on.

Re-asking only the truncated items with 4,096 tokens:

| arm | hit the cap | right with room | score with room |
|---|---|---|---|
| `finbase` | 9 of 200 | 4 | 0.965 |
| `finlong` | 7 | 2 | 0.965 |
| `shipfp8` | 6 | 3 | 0.975 |
| `parity-single` | 5 | 3 | 0.970 |
| `parity-w4a16` | 8 | 5 | 0.975 |

Every arm including the bf16 baseline has 5 to 9 of its 200 items truncated at
768 tokens. No generation was still unfinished at 4,096.

## GSQ-RCO IQ3_S against W4A16, one variable

Both arms are the same image, the same envelope and the same flags; only the
weights differ. decode tok/s, second ladder run:

| cohort | IQ3_S | W4A16 | |
|---|---|---|---|
| C1 default | 118.3 | 112.0 | **+5.6%** |
| C2 default | 202.4 | 209.2 | −3.3% |
| C8 default | 418.2 | 688.5 | **−39.3%** |
| C1 greedy | 126.9 | 119.8 | **+5.9%** |
| C2 greedy | 215.7 | 237.2 | −9.1% |
| C8 greedy | 413.0 | 711.1 | **−41.9%** |

Mean TTFT is 1.3x to 2.8x higher on IQ3_S across every cohort. The weights are
11.77 GB against 16.00 GB, 74% of the bytes.

Every GGUF number here is the IQ3_S checkpoint. This repository does not build
GSQ-RCO weights; the file was mounted read-only from elsewhere.

## Harness

Two defects, both found by serving a `.gguf`:

- `bench/run_benchmarks.sh` used `MODEL` both as the served model and as the
  bench client's `--model`, which is its tokenizer. Given a `.gguf` path the
  client read the weights file as a tokenizer and never sent a request, while
  the server answered `/health` throughout. It now derives the `hfconfig/`
  directory beside the weights, as `single-user/start_qwen.sh` does.
- The script derives `REPO` from its own location and invokes
  `"$REPO/venv/bin/vllm"`. Run from a copy elsewhere, every bench call fails
  into its log file and the rows print with empty fields, exit code 0
  throughout.
