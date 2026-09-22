#!/usr/bin/env python3
"""Needle-in-a-haystack probe against the running server.

Builds a ~TARGET_TOKENS filler context, hides a secret passcode at each given
fractional depth, asks the model for it, and reports whether the answer
contains the passcode. Complements quality_battery.py's GSM8K lane for the
"quality at depth" question on long-context KV configs.

Each depth gets its own passcode and its own opening line, so a later depth
cannot be served from the prefix cache of an earlier one: every depth pays a
full prefill, and the report carries the server's own prompt and cached token
counts. Thinking is turned off, because a reasoning parser puts a thinking
answer in `reasoning_content` and leaves `content` empty within the 32-token
budget; the probe reads `reasoning_content` as well in case the template
ignores the flag.

Usage:
    python bench/needle_test.py [target_tokens] [depth ...]
    # default: 100000 tokens, needle at 90% depth
    python bench/needle_test.py 260000 0 0.1 0.25 0.5 0.75 0.9
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _key(path):  # same convention as quality_battery.py
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(HERE, "..", "api_key.txt"))
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b")

TARGET_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
DEPTHS = [float(d) for d in sys.argv[2:]] or [0.9]

# "All work and no play makes Jack a dull boy. " is ~46 chars, ~11 tokens.
unit = "All work and no play makes Jack a dull boy. "
filler = unit * int(TARGET_TOKENS / 11)


def post(payload, timeout=3600):
    req = urllib.request.Request(
        API + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


print(f"context ~{TARGET_TOKENS} tokens, {len(DEPTHS)} depth(s)", flush=True)
rng = random.Random()
failures = 0
for DEPTH in DEPTHS:
    needle = f"{rng.randrange(10 ** 11, 10 ** 12)}"
    cut = int(len(filler) * DEPTH)
    prompt = (
        f"Run {rng.randrange(10 ** 8, 10 ** 9)}.\n\n"
        + filler[:cut]
        + f"\n\nThe secret passcode is {needle}. Remember it exactly.\n\n"
        + filler[cut:]
        + "\n\nQuestion: what is the secret passcode? Reply with the passcode only."
    )
    t0 = time.perf_counter()
    try:
        resp = post({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
    except urllib.error.HTTPError as e:
        failures += 1
        print(f"depth {DEPTH:5.0%}  HTTP {e.code}: {e.read()[:300].decode(errors='replace')}", flush=True)
        continue
    elapsed = time.perf_counter() - t0
    msg = resp["choices"][0]["message"]
    answer = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    usage = resp.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    ok = needle in answer
    failures += not ok
    print(f"depth {DEPTH:5.0%}  prompt {usage.get('prompt_tokens')}  cached {cached}  "
          f"{elapsed:6.1f}s  {'RETRIEVED' if ok else 'MISSED'}  answer: {answer.strip()[:80]!r}", flush=True)

sys.exit(1 if failures else 0)
