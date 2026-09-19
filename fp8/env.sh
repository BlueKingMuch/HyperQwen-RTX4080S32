#!/bin/bash
# fp8/env.sh — one definition of the FP8 Triton attention environment, sourced
# by both launchers right before they build their attention flags:
#
#   source "$REPO/fp8/env.sh"
#   resolve_fp8_kernels 1        # or 0
#
# fp8/install.sh puts four steps into the venv's vLLM and leaves every one of
# them off. This turns the set on together, because that is how they were
# measured: the flat prefill mapping needs full-causal, the 32-segment selector
# needs MQ3D, and V-chunked needs the 896 block size the launcher passes with
# it. Each is a plain vLLM environment variable, so exporting any of them
# yourself keeps your value -- the block below only fills in what is unset.
#
#   VLLM_TRITON_FP8_CAUSAL_FULL     full-causal FP8 attention (2D routes)
#   VLLM_TRITON_FP8_PREFILL_FLAT    flat prefill index mapping
#   VLLM_TRITON_FP8_MQ3D            FP8 multi-query 3D Split-KV
#   VLLM_TRITON_FP8_MQ3D_MIXED_TARGET  that path on the target's attention only
#   VLLM_TRITON_FP8_MQ3D_QMAX       queries per 3D block
#   VLLM_TRITON_FP8_MQ3D_SEGMENTS   parallel softmax segments, 16 or 32
#   VLLM_TRITON_FP8_V_CHUNKED       V in one 32-token tile per chunk in a block
#
# They only do anything on --attention-backend TRITON_ATTN with an fp8 KV cache;
# on any other backend the gates are false and the original loops run.

resolve_fp8_kernels() {
  local on=${1:?usage: resolve_fp8_kernels 0|1}
  case "$on" in 0|1) ;; *) echo "fp8/env.sh: FP8_KERNELS=$on (want 0 or 1)" >&2; exit 1 ;; esac
  [ "$on" = 1 ] || return 0
  # Setting a name vLLM does not know is a no-op that reads like a measurement.
  # These four come from fp8/install.sh rather than from patches/, so on a venv
  # where that step never ran they are not in the registry at all.
  if [ -x "${PY:-$REPO/venv/bin/python}" ] && ! "${PY:-$REPO/venv/bin/python}" -c \
      "import vllm.envs as e, sys; sys.exit(0 if 'VLLM_TRITON_FP8_V_CHUNKED' in e.environment_variables else 1)" 2>/dev/null; then
    echo "[fp8] WARNING: the fp8 steps are not in this vLLM (bash fp8/install.sh)." \
         "CAUSAL_FULL, PREFILL_FLAT, MQ3D_SEGMENTS and V_CHUNKED will be ignored." >&2
  fi
  export VLLM_TRITON_FP8_CAUSAL_FULL=${VLLM_TRITON_FP8_CAUSAL_FULL:-1}
  export VLLM_TRITON_FP8_PREFILL_FLAT=${VLLM_TRITON_FP8_PREFILL_FLAT:-1}
  export VLLM_TRITON_FP8_MQ3D=${VLLM_TRITON_FP8_MQ3D:-1}
  export VLLM_TRITON_FP8_MQ3D_MIXED_TARGET=${VLLM_TRITON_FP8_MQ3D_MIXED_TARGET:-1}
  export VLLM_TRITON_FP8_MQ3D_QMAX=${VLLM_TRITON_FP8_MQ3D_QMAX:-8}
  export VLLM_TRITON_FP8_MQ3D_SEGMENTS=${VLLM_TRITON_FP8_MQ3D_SEGMENTS:-32}
  export VLLM_TRITON_FP8_V_CHUNKED=${VLLM_TRITON_FP8_V_CHUNKED:-1}
  echo "[fp8] CAUSAL_FULL=$VLLM_TRITON_FP8_CAUSAL_FULL FLAT=$VLLM_TRITON_FP8_PREFILL_FLAT" \
       "MQ3D=$VLLM_TRITON_FP8_MQ3D MIXED_TARGET=$VLLM_TRITON_FP8_MQ3D_MIXED_TARGET" \
       "QMAX=$VLLM_TRITON_FP8_MQ3D_QMAX SEGMENTS=$VLLM_TRITON_FP8_MQ3D_SEGMENTS" \
       "V_CHUNKED=$VLLM_TRITON_FP8_V_CHUNKED" >&2
}

# GGUF weights need three flags the safetensors path does not, and the launcher
# can tell which it has from the model path alone: the out-of-tree plugin claims
# a model whose path ends in .gguf, and nothing else. The tokenizer and the HF
# config come from an hfconfig/ directory beside the file when there is one,
# because a GGUF carries neither in the form vLLM's tokenizer loader wants.
resolve_gguf_args() {
  local model=${1:?usage: resolve_gguf_args <model path>}
  GGUF_ARGS=()
  case "$model" in *.gguf) ;; *) return 0 ;; esac
  local dir; dir=$(dirname "$model")
  local cfg=$dir
  [ -d "$dir/hfconfig" ] && cfg=$dir/hfconfig
  GGUF_ARGS=(--quantization gguf --tokenizer "$cfg" --hf-config-path "$cfg")
  echo "[fp8] GGUF weights: $(basename "$model"), tokenizer and config from $cfg" >&2
}
