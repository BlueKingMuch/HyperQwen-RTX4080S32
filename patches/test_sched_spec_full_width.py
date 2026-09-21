"""sched-spec-full-width.patch on vLLM's own Scheduler (CPU, the tiny facebook/opt-125m config from the Hub, an
ngram speculative config of 7 tokens, budget 512): a running prefill chunk that leaves 2 / 5 / 7 tokens of the
step budget, followed by a running decode request carrying 7 drafts.

    docker run --rm --gpus all --entrypoint /app/venv/bin/python -e EXPECT=guard \
        -v "$PWD/patches:/work:ro" hyperqwen-rtx4080s32:local /work/test_sched_spec_full_width.py

Without the patch (EXPECT=trim) the decode row is scheduled with 2 / 5 / 7 tokens, a partial-width verify step;
with it (EXPECT=guard) the row is absent from the step and the solo spec row still gets its 8. --gpus all only
lets vLLM's platform detection build the config; nothing runs on the card."""
import os, sys
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
import torch

BUDGET = int(os.environ.get("BUDGET", "512"))
K = 7


def make_scheduler(max_model_len=2048):
    from vllm.engine.arg_utils import EngineArgs
    args = EngineArgs(model="facebook/opt-125m", max_model_len=max_model_len, max_num_batched_tokens=BUDGET, max_num_seqs=16,
                      enable_chunked_prefill=True, enable_prefix_caching=False, enforce_eager=True, gpu_memory_utilization=0.5,
                      speculative_config={"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_max": 4, "prompt_lookup_min": 1})
    vllm_config = args.create_engine_config()
    vllm_config.cache_config.num_gpu_blocks = 100000
    spec_kv = FullAttentionSpec(block_size=vllm_config.cache_config.block_size, num_kv_heads=1, head_size=1, dtype=torch.float16)
    bs = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(num_blocks=100000, kv_cache_tensors=[KVCacheTensor(size=1024, layers=["layer"], layer_stride=0, block_stride=0)],
                                    kv_cache_groups=[KVCacheGroupSpec(["layer"], spec_kv)])
    return Scheduler(vllm_config=vllm_config, kv_cache_config=kv_cache_config, log_stats=False, block_size=bs,
                     structured_output_manager=StructuredOutputManager(vllm_config))


def add_request(sched, rid, n_prompt, max_tokens=256):
    req = Request(request_id=rid, prompt_token_ids=list(range(1, n_prompt + 1)), sampling_params=SamplingParams(max_tokens=max_tokens),
                  pooling_params=None, arrival_time=0.0)
    sched.add_request(req)
    return req


def step_tokens(out):
    return dict(out.num_scheduled_tokens)


def run_case(prompt_len, expect):
    """A long prompt (prefill) that is already partially done sits first in `running`; a decode request with 7
    drafts sits second; the step schedules the prefill's remaining chunk first."""
    sched = make_scheduler()
    long_req = add_request(sched, "long", prompt_len)
    chat = add_request(sched, "chat", 100)
    # one step moves both into the running set; then set the state the engine would leave behind: the chat past
    # its prefill with one sampled token and 7 drafts, the long prompt with BUDGET - leftover tokens still to go
    out = sched.schedule()
    sched.running.sort(key=lambda r: r.request_id != "long")   # the long one first, as it arrived first
    # put both into a decode / mid-prefill state by hand
    chat.num_computed_tokens = 100
    chat.append_output_token_ids(7)                      # one sampled token
    chat.spec_token_ids = [11, 12, 13, 14, 15, 16, 17]   # 7 drafts for the next step
    long_req.num_computed_tokens = prompt_len - (BUDGET - expect["leftover"])
    from vllm.v1.request import RequestStatus
    long_req.status = RequestStatus.RUNNING
    chat.status = RequestStatus.RUNNING
    sched.running = [long_req, chat]
    for r in (long_req, chat):
        try:
            sched.waiting.remove_request(r)
        except Exception:
            pass
    out = sched.schedule()
    got = step_tokens(out)
    return got


if __name__ == "__main__":
    ok = True
    for leftover in (2, 5, 7):
        got = run_case(prompt_len=BUDGET * 3, expect={"leftover": leftover})
        long_t = got.get("long", 0); chat_t = got.get("chat", None)
        print(f"budget leftover {leftover}: long chunk {long_t}, chat scheduled tokens {chat_t}")
        # before the guard: chat_t == leftover (trimmed); with the guard: chat absent (skipped)
        if os.environ.get("EXPECT") == "guard":
            ok &= chat_t is None and long_t == BUDGET - leftover
        elif os.environ.get("EXPECT") == "trim":
            ok &= chat_t == leftover
    # untrimmed: a decode row alone gets its full 8
    sched = make_scheduler()
    chat = add_request(sched, "solo", 100)
    sched.schedule()
    chat.num_computed_tokens = 100; chat.append_output_token_ids(7); chat.spec_token_ids = list(range(11, 18))
    from vllm.v1.request import RequestStatus
    chat.status = RequestStatus.RUNNING; sched.running = [chat]
    got = step_tokens(sched.schedule())
    print("solo spec row:", got)
    ok &= got.get("solo") == 8
    print("RESULT", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
