#!/usr/bin/env bash
set -euo pipefail

# Install the FP8 Triton attention steps into this repo's vLLM venv, in the
# order ./series gives. The first three are not patch files: they rewrite
# installed sources by exact anchor matching, guard themselves with sha256 pins
# of what they read, and prove their output reverses byte-for-byte back to its
# parent. The last step is two ordinary patches, and those carry their pins in
# fp8-paged/PINS.
#
#   bash fp8/install.sh                      # gate, install, gate again
#   bash fp8/install.sh --check              # verify what this repo carries, write nothing
#   bash fp8/install.sh --write-sums         # re-cut SHA256SUMS after a repin
#   FP8_PINS=write bash fp8/install.sh       # install and rewrite fp8-paged/PINS
#
# The pins in these files are content-hashed by SHA256SUMS, so anything that
# rewrites a pin -- scripts/repin-fp8-installers.py, or FP8_PINS=write rewriting
# fp8-paged/PINS -- leaves that gate stale. --write-sums is how it is re-cut; it
# is the one mode that does not check the gate first.
#
# Every step runs AFTER patches/series and AFTER kvarn/install.sh, because all
# of them pin bytes of files those two touch. The order lives in ./series.
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
#
# When a pin does not match, the tree under it moved. Do not edit the pin:
# run scripts/repin-fp8-installers.py against the tree and read what moved.

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
if [ "${1:-}" = "--write-sums" ]; then
  ( cd "$HERE" && find . -type f ! -name 'SHA256SUMS*' ! -name install.sh -printf '%P\n' \
      | sort | while IFS= read -r f; do printf '%s *%s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f"; done \
      > SHA256SUMS.new && mv SHA256SUMS.new SHA256SUMS )
  echo "fp8/SHA256SUMS re-cut over $(grep -c . "$HERE/SHA256SUMS") files"
  exit 0
fi

# Where the steps write the sources they replace and the manifests that pin
# them. /opt is right inside the image; a venv install on a machine where /opt is
# not writable sets FP8_ARCHIVE to somewhere it owns. verify.sh reads the same
# variable, so both have to agree.
ARCHIVE=${FP8_ARCHIVE:-/opt/fp8}
if [ "${1:-}" != "--check" ] && [ -e "$ARCHIVE" ]; then
  echo "ERROR: $ARCHIVE already exists." >&2
  echo "       Each step refuses to overwrite an archive, so this run would abort" >&2
  echo "       part-way. The steps are not idempotent: they rewrite installed" >&2
  echo "       sources once, from a parent they pin. To reinstall, start from a" >&2
  echo "       clean vLLM tree and remove $ARCHIVE; to check an existing install," >&2
  echo "       run verify.sh. FP8_ARCHIVE moves the archive elsewhere." >&2
  exit 1
fi
if [ "${1:-}" != "--check" ]; then
  mkdir -p "$ARCHIVE" 2>/dev/null || {
    echo "ERROR: cannot create $ARCHIVE. Set FP8_ARCHIVE to a writable path." >&2; exit 1; }
  rmdir "$ARCHIVE" 2>/dev/null || true
fi

say() { printf '== %s\n' "$*"; }

say "what this repo carries"
( cd "$HERE" && sha256sum -c SHA256SUMS >/dev/null ) \
  || { echo "ERROR: fp8/SHA256SUMS does not match the files here." >&2; exit 1; }
echo "   SHA256SUMS: OK"

SERIES=()
while IFS= read -r name; do SERIES+=("$name"); done \
  < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series")
[ "${#SERIES[@]}" = 4 ] || { echo "ERROR: fp8/series lists ${#SERIES[@]} steps, not 4" >&2; exit 1; }
echo "   series: ${SERIES[*]}"

if [ "${1:-}" = "--check" ]; then
  "$PY" - "$HERE" <<'PY'
import ast, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(root.rglob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf8'), str(path))
print("   %d python files parse" % len(files))
PY
  echo "fp8: inputs OK (--check: nothing installed)"
  exit 0
fi

SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] || { echo "cannot import vllm with $PY (README: Setup)" >&2; exit 1; }
SITE=$(dirname "$SP")
K=v1/attention/ops/triton_unified_attention.py
H=v1/attention/ops/triton_attention_helpers.py

# A stage of fp8-paged/PINS, checked against the tree or rewritten from it.
pins() {
  local stage=$1 line file want have
  while read -r _ file want; do
    have=$(sha256sum "$SP/$file" | cut -d' ' -f1)
    if [ "${FP8_PINS:-}" = write ]; then
      printf '%-17s %-50s %s\n' "$stage" "$file" "$have" >> "$HERE/fp8-paged/PINS.new"
    elif [ "$have" != "$want" ]; then
      echo "ERROR: $stage: $file is $have, not the pinned $want" >&2
      echo "       The tree under fp8-paged moved. Regenerate: FP8_PINS=write bash fp8/install.sh" >&2
      exit 1
    fi
  done < <(grep "^$stage " "$HERE/fp8-paged/PINS")
}

# Keep the header, drop every stage line: they are about to be re-measured.
[ "${FP8_PINS:-}" = write ] && { sed -n '/^#/p' "$HERE/fp8-paged/PINS" > "$HERE/fp8-paged/PINS.new"; }

for step in "${SERIES[@]}"; do
  case "$step" in
    fp8-causal)
      say "$step"
      ( cd "$HERE/fp8-causal" && "$PY" -B test_fp8_causal_cpu.py --parent "$SP/$K" --helper "$SP/$H" >/dev/null )
      echo "   CPU gate: OK"
      ( cd "$HERE/fp8-causal" && "$PY" -B install_fp8_causal.py \
          --vllm-root "$SP" --archive "$ARCHIVE/causal/parent" >/dev/null )
      ( cd "$HERE/fp8-causal" && "$PY" -B test_fp8_causal_adapters_cpu.py >/dev/null )
      echo "   installed, adapters: OK"
      ;;
    fp8-causal-r2)
      say "$step"
      ( cd "$HERE/fp8-causal/r2" && "$PY" -B test_fp8_causal_r2_cpu.py --r1-source "$SP/$K" >/dev/null )
      echo "   CPU gate: OK"
      ( cd "$HERE/fp8-causal/r2" && "$PY" -B install_fp8_causal_r2.py \
          --vllm-root "$SP" --archive "$ARCHIVE/causal" >/dev/null )
      ( cd "$HERE/fp8-causal/r2" && "$PY" -B test_fp8_causal_r2_adapters_cpu.py >/dev/null )
      echo "   installed, adapters: OK"
      ;;
    fp8-composite)
      say "$step"
      ( cd "$HERE/fp8-composite" && "$PY" -B test_composite_cpu.py \
          --vllm-root "$SP" --causal-archive "$ARCHIVE/causal" >/dev/null )
      echo "   CPU gate: OK"
      ( cd "$HERE/fp8-composite" && "$PY" -B install_composite.py \
          --vllm-root "$SP" --archive "$ARCHIVE/composite" --causal-archive "$ARCHIVE/causal" >/dev/null )
      ( cd "$HERE/fp8-composite" && "$PY" -B test_composite_cpu.py --installed \
          --vllm-root "$SP" --archive "$ARCHIVE/composite" --causal-archive "$ARCHIVE/causal" >/dev/null )
      echo "   installed, seal: OK"
      ;;
    fp8-paged)
      say "$step"
      pins tile-ptrs.before
      patch -p1 --dry-run --batch --fuzz 0 -d "$SP" < "$HERE/fp8-paged/triton-fp8-paged-tile-ptrs.patch" >/dev/null
      patch -p1 --batch --fuzz 0 -d "$SP" < "$HERE/fp8-paged/triton-fp8-paged-tile-ptrs.patch" >/dev/null
      pins tile-ptrs.after
      grep -Fq 'def _paged_tile_ptrs(' "$SP/$K"
      [ "$(grep -c 'physical_block_idx\[:, None\] \* stride_v_cache_0' "$SP/$K")" = 2 ]
      TRITON_INTERPRET=1 "$PY" -B "$HERE/fp8-paged/test_paged_tile_offsets_cpu.py" \
        --interpret --candidate "$SP/$K" >/dev/null
      echo "   tile-ptrs: OK"

      pins v-chunked.before
      patch -p1 --dry-run --batch --fuzz 0 -d "$SP" < "$HERE/fp8-paged/triton-fp8-v-chunked-b896.patch" >/dev/null
      patch -p1 --batch --fuzz 0 -d "$SP" < "$HERE/fp8-paged/triton-fp8-v-chunked-b896.patch" >/dev/null
      pins v-chunked.after
      grep -Fq 'V_CHUNKED: tl.constexpr' "$SP/$K"
      grep -Fq 'block_stride_v' "$SP/v1/attention/ops/triton_reshape_and_cache_flash.py"
      grep -Fq 'def _fp8_chunked_caches(' "$SP/v1/attention/backends/triton_attn.py"
      grep -Fq '"VLLM_TRITON_FP8_V_CHUNKED"' "$SP/envs.py"
      [ "$(grep -c 'block_size in (880, 896)' "$SP/$K")" = 1 ]
      [ "$(grep -c 'block_size in (880, 896)' "$SP/v1/attention/backends/triton_attn.py")" = 1 ]
      TRITON_INTERPRET=1 "$PY" -B "$HERE/fp8-paged/test_fp8_v_chunked_cpu.py" \
        --interpret --backend --candidate "$SITE" >/dev/null
      echo "   v-chunked-b896: OK"
      ;;
    *) echo "ERROR: fp8/series names an unknown step: $step" >&2; exit 1 ;;
  esac
done

if [ "${FP8_PINS:-}" = write ]; then
  mv "$HERE/fp8-paged/PINS.new" "$HERE/fp8-paged/PINS"
  echo "   fp8-paged/PINS rewritten from this tree"
fi

find "$SP" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
"$PY" -c 'import vllm.envs as e
for name, default in (("VLLM_TRITON_FP8_CAUSAL_FULL", False), ("VLLM_TRITON_FP8_PREFILL_FLAT", 0),
                      ("VLLM_TRITON_FP8_MQ3D_SEGMENTS", 16), ("VLLM_TRITON_FP8_V_CHUNKED", False)):
    assert getattr(e, name) == default, (name, getattr(e, name), default)
print("   four flags registered, all at their off default")'
echo "fp8 installed"
