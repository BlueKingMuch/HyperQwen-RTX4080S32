# Reproduction: the int8 prefill on the GGUF tiles, and recall at the full window

Same box as [ada-029-fp8-gguf.md](ada-029-fp8-gguf.md), one arm, one variable:
the parent commit against the commit that moved the prefill chunk's GEMM onto
the tiles (`gguf-plugin/README.md`, "The wide forms"). Both images built from
this repository, the same `.env`, the same probes, the same session.

[← back to the reproductions index](README.md)

## Hardware and stack

| | |
|---|---|
| GPU | RTX 4080 SUPER, 32,760 MiB, 249.6 W cap, undervolted |
| OS | Windows 11 Pro 26200, WSL2, Docker |
| vLLM | 0.29.0, torch 2.13.0+cu130, Triton 3.7.1 |
| image | `docker compose build single`; `verify.sh --install` passes with 0 failures |
| weights | a 3.84 bpw GGUF of the 27B model, 866 tensors over 11 quantised types |
| drafter | DFlash2 W4A16, 7 draft tokens, int8 per-token-head KV |
| `.env` | `CTX=int4`, `SPEC=dflash2`, `MAX_SEQS=8`, `GPU_UTIL=0.88`, `PREFIX_CACHE=1` |
| window | `max_model_len` 262,144, `max_num_batched_tokens` 2,048 |

The weights are a per-tensor mix: IQ4_XS 27.5 % of the bytes, IQ3_S 27.2 %,
IQ3_XXS 18.8 %, Q4_K 12.7 %, Q6_K 8.3 %, Q3_K 2.9 %, Q5_K 1.6 %, then IQ2_XXS,
IQ2_XS, Q8_0 and one Q2_K tensor. The first three and Q6_K, Q3_K, Q8_0, IQ2_XXS
and IQ2_XS take the wide form; Q4_K, Q5_K and IQ2_S keep the dequantise path.
The file is not produced by this repository and is mounted read-only.

## KV pool

| | tokens at a 262,144 window |
|---|---|
| parent | 606,515 |
| wide form | 610,029 |

## Prefill

`bench/labd_bench.py <tag> --ctx 100000`, six tasks over one ~100,500-token
document. The first task pays the cold prefill; the other five hit the prefix
cache and recompute one chunk. `ttft` in seconds.

| task | parent | wide form |
|---|---|---|
| copy (cold) | 90.02 | 71.10 |
| code | 3.11 | 2.41 |
| edit | 3.09 | 2.40 |
| quote | 3.10 | 2.41 |
| summary | 3.10 | 2.41 |
| qa | 3.10 | 2.40 |

A second probe on a 100,015-token document, `max_tokens` 8 on the cold request:
90.40 s against 69.62 s cold, and the cached follow-ups 2.25-2.26 s against
1.81-1.89 s at 8 output tokens, 6.17-6.42 s against 5.64-5.67 s at 512.

## Decode

Same runs, `decode` is concurrency over mean TPOT and `tok/step` is the
drafter's accepted tokens per step.

| task | parent decode | wide decode | parent tok/step | wide tok/step |
|---|---|---|---|---|
| copy | 182.4 | 186.8 | 7.73 | 7.73 |
| code | 66.6 | 68.4 | 2.79 | 2.75 |
| edit | 42.6 | 43.7 | 1.79 | 1.79 |
| quote | 56.1 | 48.1 | 2.38 | 1.98 |
| summary | 59.7 | 54.0 | 2.52 | 2.23 |
| qa | 51.6 | 52.2 | 2.18 | 2.15 |

At `--ctx 4000` the copy task reads 245.4 against 255.1 tok/s at 7.83 against
7.86 tok/step, and the six-task total 90.6 against 93.2.

`tok/step` differs per task between the two runs, so labd's aggregate rate is
not a single-variable comparison of the decode kernel. The kernel-level A/B is
`gguf-plugin/test_gluon_kq_gpu.py --time`: over its twenty shapes at 8 rows,
normalised by its own Q4_K control, the median ratio is 0.999, and the largest
shape alternated parent/child three times each reads 858/863/861 µs against
860/860/854.

## GSM8K

`bench/quality_battery.py <tag> --gsm-only --gsm-n 200`, one run each: 0.970
parent, 0.965 wide form. The repository's published band across every shipped
configuration is 0.950 to 0.965 at n=200 with a stated standard error of about
1.3 points (`docs/gotchas.md`, 40).

## Recall at the full window

`bench/needle_test.py 260000 0 0.1 0.25 0.5 0.75 0.9` on the wide form. Each
depth carries its own passcode and its own opening line, so none is served from
the prefix cache of an earlier one: `cached` is 0 throughout and every depth
pays a full prefill.

| depth | prompt tokens | cached | prefill | answer |
|---|---|---|---|---|
| 0 % | 260,062 | 0 | 288.0 s | retrieved |
| 10 % | 260,063 | 0 | 288.1 s | retrieved |
| 25 % | 260,063 | 0 | 289.4 s | retrieved |
| 50 % | 260,063 | 0 | 290.5 s | retrieved |
| 75 % | 260,063 | 0 | 289.3 s | retrieved |
| 90 % | 260,064 | 0 | 287.4 s | retrieved |

The window is 262,144, so each prompt sits about 2,080 tokens below it. The same
six depths with a natural-language filler instead of the repeated sentence, and
a different passcode at each: 260,072 to 260,075 tokens, 289.6 to 293.2 s,
retrieved in all six. Twelve prompts, twelve passcodes, no miss and no partial
answer.

The same probe at 100,057 tokens and 90 % depth retrieves in 71.4 s.

## Reproducing it

```
docker compose up -d single
docker exec hyperqwen-rtx4080s32-single-1 /app/venv/bin/python \
  /app/bench/needle_test.py 260000 0 0.1 0.25 0.5 0.75 0.9
docker exec hyperqwen-rtx4080s32-single-1 /app/venv/bin/python \
  /app/bench/labd_bench.py run --ctx 100000 --corpus /cache/bench/labd_corpus_long.txt
docker exec hyperqwen-rtx4080s32-single-1 /app/venv/bin/python \
  /app/bench/quality_battery.py run --gsm-only --gsm-n 200
```

Four things to know before repeating it:

- `bench/needle_test.py` sent no `enable_thinking: False`. With a reasoning
  parser the answer lands in `reasoning_content`, `content` comes back empty
  inside the 32-token budget, and the probe reports MISSED against a model that
  retrieves. It now turns thinking off, reads `reasoning_content` as a
  fallback, takes several depths in one invocation, gives each its own passcode
  and opening line, and prints the server's own prompt and cached token counts.
- `bench/quality-data/` is in both `.dockerignore` and `.gitignore`, so the
  GSM8K parquet is neither in the repository nor in the image. It has to be
  copied into the container before `--gsm-only` runs, or pyarrow raises
  `FileNotFoundError` on the path inside.
- A `torch_aot_compile` entry in the `/cache` volume survived the rebuild and
  killed the engine in its dummy run with `assert_size_stride ... expected size
  256==512`; compose restarts the container, so it loops. Moving that one
  directory aside let the same image boot and serve. `docs/gotchas.md` 12
  records the same failure mode from a different trigger.
- The first prefill after a cold Triton cache compiles the wide variants the
  model's layers need, twelve for this file, and takes about 45 s once. The
  server logs one `tiles_grouped_kernel` JIT warning; prefills of the same
  shape afterwards run in 1.8-2.1 s. The startup profile run does not reach the
  wide form, so the cost lands on the first request.
