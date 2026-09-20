#!/usr/bin/env bash
set -euo pipefail

# Install the FP8 Triton attention steps into this repo's vLLM venv, in the
# order ./series gives. All four are patch files, applied with --fuzz 0: a hunk
# whose context has moved fails the build by name instead of landing by guess.
#
#   bash fp8/install.sh             # apply the four steps
#   bash fp8/install.sh --check     # the files this repo carries, install nothing
#
# Every step runs AFTER patches/series and AFTER kvarn/install.sh, because all
# of them touch files those two touch. The order lives in ./series and is not a
# preference: each step's context is what the one before it wrote.
#
# What the four steps put behind which flag. All default off, and none of them
# changes a default path:
#
#   fp8-causal        VLLM_TRITON_FP8_CAUSAL_FULL    full-causal FP8, 2D and 3D
#   fp8-causal-r2     (the same flag)                2D only; every 3D route
#                                                    keeps the original loop
#   fp8-composite     VLLM_TRITON_FP8_PREFILL_FLAT   flat prefill index mapping
#                     VLLM_TRITON_FP8_MQ3D_SEGMENTS  32 softmax segments (16)
#   fp8-paged         VLLM_TRITON_FP8_V_CHUNKED      V in one 32-token tile per
#                                                    chunk inside a KV block

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY=${PY:-$REPO/venv/bin/python}
# Outside the repo's venv -- a --check run on a machine that has no install --
# take the first interpreter that is actually there and runs.
if [ ! -x "$PY" ]; then
  for c in python3 python; do
    command -v "$c" >/dev/null 2>&1 && "$c" -c '' 2>/dev/null && { PY=$c; break; }
  done
fi

say() { printf '== %s\n' "$*"; }

# step -> patch files, in apply order within the step.
patches_for() {
  case "$1" in
    fp8-causal)     echo "fp8-causal/fp8-causal.patch" ;;
    fp8-causal-r2)  echo "fp8-causal/r2/fp8-causal-r2.patch" ;;
    fp8-composite)  echo "fp8-composite/fp8-composite.patch" ;;
    fp8-paged)      echo "fp8-paged/triton-fp8-paged-tile-ptrs.patch"
                    echo "fp8-paged/triton-fp8-v-chunked-b896.patch" ;;
    *) echo "ERROR: fp8/series names an unknown step: $1" >&2; exit 1 ;;
  esac
}

SERIES=()
while IFS= read -r name; do SERIES+=("$name"); done \
  < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series")
[ "${#SERIES[@]}" = 4 ] || { echo "ERROR: fp8/series lists ${#SERIES[@]} steps, not 4" >&2; exit 1; }
say "series: ${SERIES[*]}"

for step in "${SERIES[@]}"; do
  while IFS= read -r rel; do
    [ -f "$HERE/$rel" ] || { echo "ERROR: $step names $rel, which is not here" >&2; exit 1; }
  done < <(patches_for "$step")
done

# Every patch of every step, one per line.
all_patches() { for step in "${SERIES[@]}"; do patches_for "$step"; done; }

if [ "${1:-}" = "--check" ]; then
  echo "   $(all_patches | wc -l) patch files present for ${#SERIES[@]} steps"
  # If a vLLM tree is reachable, say whether the steps are in it. A reverse dry
  # run at --fuzz 0 succeeds only against the tree that patch produced; where a
  # later step rewrote the same lines, the content check settles it.
  sp=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
  if [ -n "$sp" ] && [ -d "$sp" ]; then
    miss=0
    while IFS= read -r rel; do
      patch -p1 -R --dry-run -s --fuzz 0 -d "$sp" < "$HERE/$rel" >/dev/null 2>&1 && continue
      "$PY" "$REPO/patches/_check_applied.py" "$HERE/$rel" "$sp" >/dev/null 2>&1 && continue
      echo "   NOT applied: $rel" >&2; miss=$((miss+1))
    done < <(all_patches)
    [ "$miss" = 0 ] || { echo "fp8: $miss of $(all_patches | wc -l) steps are not in $sp" >&2; exit 1; }
    echo "   all steps applied in $sp"
  else
    echo "   no vLLM tree reachable; checked the files only"
  fi
  echo "fp8: OK (--check: nothing installed)"
  exit 0
fi

SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] || { echo "cannot import vllm with $PY (README: Setup)" >&2; exit 1; }
K=v1/attention/ops/triton_unified_attention.py

for step in "${SERIES[@]}"; do
  say "$step"
  while IFS= read -r rel; do
    patch -p1 --batch --fuzz 0 -d "$SP" < "$HERE/$rel"
  done < <(patches_for "$step")
done

# Structural checks on the finished tree. --fuzz 0 already refused any hunk
# whose context had moved, and each step's context is the previous step's
# output, so a mid-sequence assertion would say nothing these do not. They name
# themselves when they fail: under `set -e` a bare `grep -q` aborts wordlessly,
# which is not a diagnosis.
fail_at() { echo "ERROR: $1: $2" >&2
            echo "       The steps applied but did not land where expected." >&2; exit 1; }
# present at all
have() { grep -qF -- "$2" "$SP/$1" || fail_at "$1" "not found: $2"; }
# present exactly N times -- the two guards that say the gate was not duplicated
count() { local n; n=$(grep -cF -- "$2" "$SP/$1")
          [ "$n" = "$3" ] || fail_at "$1" "expected $3 of \"$2\", found $n"; }
have "$K" 'def _paged_tile_ptrs('
have "$K" 'V_CHUNKED: tl.constexpr'
have v1/attention/ops/triton_reshape_and_cache_flash.py 'block_stride_v'
have v1/attention/backends/triton_attn.py 'def _fp8_chunked_caches('
have envs.py '"VLLM_TRITON_FP8_V_CHUNKED"'
count "$K" 'block_size in (880, 896)' 1
count v1/attention/backends/triton_attn.py 'block_size in (880, 896)' 1
echo "   structure: OK"

find "$SP" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
# The live registry, not the source: a hunk can land and the flag still not
# register.
"$PY" -c 'import vllm.envs as e
for name, default in (("VLLM_TRITON_FP8_CAUSAL_FULL", False), ("VLLM_TRITON_FP8_PREFILL_FLAT", 0),
                      ("VLLM_TRITON_FP8_MQ3D_SEGMENTS", 16), ("VLLM_TRITON_FP8_V_CHUNKED", False)):
    assert getattr(e, name) == default, (name, getattr(e, name), default)
print("   four flags registered, all at their off default")'
echo "fp8 installed"
