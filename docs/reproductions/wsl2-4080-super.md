# Reproduction: RTX 4080 SUPER 32 GB, Windows 11 / WSL2, Docker

Setup B on an Ada card with 32 GB, from an image built out of this repository
rather than pulled.

[← back to the reproductions index](README.md)

## Hardware and stack

| | |
|---|---|
| GPU | RTX 4080 SUPER, 32,760 MiB, driver 616.56, 250 W cap, undervolted |
| OS | Windows 11 Pro 26200, WSL2, Docker 29.7.2 |
| image | `docker compose build` at `c0c81bb`; `verify.sh --install` passes with 0 failures, all of `patches/` plus KVarN |
| vLLM | 0.28.0, torch 2.13.0+cu130, Triton 3.7.1 |
| model | `Qwen3.8-27B-W4A16-AutoRound-fast` with the DFlash2 drafter |
| setup | **B** — `.env` left at its defaults (`SPEC=dflash2`, `PREFIX_CACHE=1`, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`), `--profile single` |
| KV pool | 68,605 tokens at `max_model_len` 65,536 |

## `bench/run_benchmarks.sh single`

Second run after a start, as the protocol asks.

| cohort | greedy e2e | greedy decode | `T=default` e2e | `T=default` decode |
|---|---|---|---|---|
| C1 | 108.80 | 112.6 | 108.32 | 112.5 |
| C2 | 201.61 | 232.8 | 185.02 | 212.1 |
| C4 | 315.11 | 380.2 | 311.47 | 374.9 |
| C8 | 289.88 | 500.9 | 308.79 | 447.9 |

`tok/step`, greedy: 3.15 (C1), 3.54 (C2), 3.23 (C4), 3.17 (C8).

Mean TTFT, greedy: 187.79 ms (C1), 340.72 ms (C2), 408.06 ms (C4), 2611.59 ms
(C8).

For comparison, [docs/docker.md](../docker.md) records 112.6 e2e and 115.7
decode for the 3090 in the same container at default sampling. Memory
bandwidth is 736 GB/s here against the 3090's 936.

## Quality

`bench/quality_battery.py <tag> --gsm-only`, n=200:

| | |
|---|---|
| GSM8K | **0.965** |
| mean tokens | 379 |
| wall clock | 118 s |

## `pyarrow` is missing from the image build

The quality battery cannot run on a locally built image:

```
  File "/app/bench/quality_battery.py", line 27, in <module>
    import pyarrow.parquet as pq
ModuleNotFoundError: No module named 'pyarrow'
```

`bench/quality_battery.py`, `bench/act_calib.py`, `drafter/collect_prompts.py`
and `prepare/build_draft_vocab.py` all import `pyarrow.parquet`, and the
datasets they read ship as parquet. It is not in `docker/requirements.txt`, and
`pandas==3.0.5` does not pull it in — `pip list` in a freshly built image shows
pandas without it.

The published image carries it. [docs/docker.md](../docker.md) offers
`docker compose build` as the alternative to pulling; the field-test protocol in
the README asks for the battery. `docker/requirements.txt` in this fork now pins
`pyarrow==21.0.0`; the GSM8K number above was measured with it installed by hand
into a running container.
