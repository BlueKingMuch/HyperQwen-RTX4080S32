# Same stack as the README's venv install, frozen: Python 3.12 venv at /app/venv,
# vLLM 0.29.0 (torch 2.13 / cu130 / Triton 3.7.1), every compatible patch in
# patches/ applied, the KVarN KV cache installed, the four FP8 Triton attention
# steps of fp8/ installed, the GGUF plugin of gguf-plugin/ fetched, patched and
# built for sm_89 and sm_120, verify.sh --install run at build time.
#
# Everything fp8/ and gguf-plugin/ add is off until an environment variable or a
# .gguf model path turns it on; .env and docs/docker.md say which.
#
# The base image is CUDA "base" + nvcc, not "devel": vLLM's wheels bring their own
# CUDA libraries, but FlashInfer JIT-compiles its fp8-KV attention kernel with nvcc
# on first use (batch mode, CTX=long) and Triton needs a C compiler for its launchers.
# The compiled kernels and the torch.compile cache live in the /cache volume, so
# that only happens once.
#
#   docker compose --profile single up -d      (see docs/docker.md)
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
      build-essential patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.12 -m venv venv && venv/bin/pip install --upgrade pip
COPY docker/requirements.txt docker/requirements.txt
RUN venv/bin/pip install -r docker/requirements.txt

COPY . .
# Patch apply order lives in patches/series: a few patches carry hunk context
# that an earlier patch adds, so the glob order of patches/*.patch is wrong.
#
# Then, in this order and for a reason:
#   kvarn/install.sh       hunks cut against the tree the whole series leaves
#   fp8/install.sh         four steps that pin the bytes the two above leave
#   gguf-plugin/install.sh a separate package, so last; it patches nothing here
RUN set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' patches/series | \
    while IFS= read -r name; do \
      case "$name" in \
        dflash2-backport.patch) echo "== skip $name (DFlash2 is native since vLLM 0.28.0)"; continue ;; \
      esac; \
      echo "== $name"; patch -p1 --fuzz 0 --no-backup-if-mismatch -d "$SP" < "patches/$name"; \
    done; \
    bash kvarn/install.sh; \
    bash fp8/install.sh; \
    bash gguf-plugin/install.sh; \
    ( cd gguf-plugin && ../venv/bin/python test_gguf_rco_cpu.py --names iq3s-tensor-names.json ); \
    bash verify.sh --install

# HOME is a volume: torch.compile cache (~/.cache/vllm), Triton (~/.triton),
# FlashInfer JIT (~/.cache/flashinfer), HF hub cache.
RUN mkdir -p /cache /app/models && chmod 1777 /cache
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_XET_HIGH_PERFORMANCE=1
VOLUME ["/cache", "/app/models"]
EXPOSE 18020
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["single"]
