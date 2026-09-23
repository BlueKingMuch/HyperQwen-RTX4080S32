#!/usr/bin/env bash
set -euo pipefail

# Fetch the out-of-tree GGUF plugin at a pinned commit, patch it, and build its
# extension for this card. Nothing of the plugin's source lives in this repo:
# the archive is fetched and its sha256 checked, so what is carried here is only
# what is ours -- the eleven patches, the Gluon decode kernels, and the CPU gate.
#
# The same shape as kvarn/install.sh, run after the vLLM series rather than
# beside it, because the plugin is installed as a package and not patched into
# the vllm one.
#
#   bash gguf-plugin/install.sh            # fetch, patch, build, install
#   bash gguf-plugin/install.sh --check    # verify what is carried here, no network
#
# Three hash gates, and they answer different questions:
#
#   SHA256SUMS         what this repo carries is what was reviewed
#   the archive        upstream's bytes at PIN are the bytes that were patched
#   before / after     the files the patches touch entered and left in the
#                      exact states the series was cut for -- a patch that lands
#                      somewhere plausible but wrong fails here, not in a kernel
#
# Regenerate nothing by hand. A patch file here is content-hashed by SHA256SUMS
# and is a derivative work of the plugin's Apache-2.0 source; fix the source of
# the change and re-export.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
# The interpreter this repo installs into. kvarn/install.sh resolves it the same
# way: without it, a bare `python3` puts the plugin in the system interpreter
# while vLLM lives in /app/venv, and `import vllm_gguf_plugin` then fails for
# the server that needs it.
PY=${PY:-$REPO/venv/bin/python}
# Outside the repo's venv -- a --check run on a machine that has no install --
# take the first interpreter that is actually there and runs.
if [ ! -x "$PY" ]; then
  for c in python3 python; do
    command -v "$c" >/dev/null 2>&1 && "$c" -c '' 2>/dev/null && { PY=$c; break; }
  done
fi

PIN=d4c1f0d082fc7cd4350da56689109a01c1f29d6c
ARCHIVE_URL="https://github.com/vllm-project/vllm-gguf-plugin/archive/${PIN:0:12}.tar.gz"
ARCHIVE_SHA=c225ff0a282e9703b924084a5b9af3a834e5882ad733ffee64027f6f1b3a755c

# sm_89 is this card. sm_120 is here because SASS is not forward-compatible
# across generations and only PTX is, so a Blackwell clone of this recipe would
# otherwise JIT every kernel on first use.
: "${TORCH_CUDA_ARCH_LIST:=8.9;12.0}"
: "${MAX_JOBS:=8}"

# The files the series touches, before it runs and after. The four added by
# gguf-dflash2-draft-gguf.patch have no BEFORE entry: params.py, config.py
# and weights_adapter/__init__.py are only edited by it, and dflash.py is new.
BEFORE="8c80ecb5fadecf60c603274ceeb3cd84500a8c6f915ea52b86d307e4775f2b2f  vllm_gguf_plugin/loader.py
a49b946f6374aa12a14a6517cd2f0140271818e59881a87f0238024b211f7984  vllm_gguf_plugin/config_parser.py
f46667f8c09a7ca8b90fd4b7c89a5eca54cca99372762c3c2853870874d6e829  vllm_gguf_plugin/csrc/gguf/mmvq.cuh
4da2c6a6c0f9b0e60661f8334f2d356322f6f4f443b51dcfa570c4da57291e38  vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu
990b6aca44615eaf39a3fa76b65992e1101cf1c4d27ec3a36070074f9f83136b  vllm_gguf_plugin/csrc/gguf/ggml-common.h
b40556bc026632e610adaee2855d42ddc3d00d99ba07fdfee7cae171ec180efe  vllm_gguf_plugin/quantization/linear.py"

AFTER="be71ead0f5c8539b89bba8ff7eb186c40a583ca29b96619c67af71241f631408  vllm_gguf_plugin/loader.py
bb4ed268b66f58393d0872a27dd07d00e40b7e9e6ae4b1213fcfe1c8fffa9e49  vllm_gguf_plugin/config_parser.py
62c48aecab78a5ef724441696f9f849e4a20e25ac1e2ae5c3f773b835f36964b  vllm_gguf_plugin/csrc/gguf/mmvq.cuh
77ce6a702daf92e78983d2e0f73d1908a689aaa0a0673ef835a234bbe9a5f7d4  vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu
e075e3e7bc0c195d7ffef2c765b0f811da8cbd04906bb2572c272decb2b20bbc  vllm_gguf_plugin/csrc/gguf/ggml-common.h
b3f7daf879cd486fb12382cd5b377104f488066b39fc0a0b4783657f486b8bc1  vllm_gguf_plugin/quantization/linear.py
73554097ab1f1efb053d5b668b8568befbc079cf512902e1cd0b7a782d762304  vllm_gguf_plugin/quantization/params.py
909da463f281bb0c39ebbbba89719562356ebb9d0efc046c8e0f6959cc0d28c1  vllm_gguf_plugin/quantization/config.py
5cdacfd6b2144f06d60faafe7ce0e67ae875cbf38b8300aefeb6507922abe44d  vllm_gguf_plugin/weights_adapter/__init__.py
d7ea8f323420ca3679d2c9c1d2badeda51f0b13f0a3fad891e99e29157a83fcb  vllm_gguf_plugin/weights_adapter/dflash.py"

say() { printf '== %s\n' "$*"; }

say "what this repo carries"
( cd "$HERE" && sha256sum -c SHA256SUMS >/dev/null ) \
  || { echo "ERROR: gguf-plugin/SHA256SUMS does not match the files here." >&2; exit 1; }
echo "   SHA256SUMS: OK"

# ./series is the single source of truth for the apply order. It and the
# directory must agree exactly, in both directions.
SERIES=()
while IFS= read -r name; do SERIES+=("$name"); done \
  < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series")
ON_DISK=$(for f in "$HERE"/patches/*.patch; do basename "$f"; done | sort)
IN_SERIES=$(printf '%s\n' "${SERIES[@]}" | sort)
[ "$ON_DISK" = "$IN_SERIES" ] || {
  echo "ERROR: gguf-plugin/series and gguf-plugin/patches/ disagree:" >&2
  comm -3 <(printf '%s\n' "$ON_DISK") <(printf '%s\n' "$IN_SERIES") | sed 's/^/    /' >&2
  exit 1
}
echo "   series and patches/ agree on ${#SERIES[@]} patches"

if [ "${1:-}" = "--check" ]; then
  echo "gguf-plugin: inputs OK (--check: nothing fetched, nothing built)"
  exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

say "the plugin's own dependencies"
"$PY" -m pip install --no-cache-dir --no-deps --require-hashes -r "$HERE/requirements.txt"

say "upstream at ${PIN:0:12}"
"$PY" - "$ARCHIVE_URL" "$WORK/plugin.tar.gz" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
echo "$ARCHIVE_SHA  $WORK/plugin.tar.gz" | sha256sum -c - >/dev/null \
  || { echo "ERROR: the fetched archive is not the pinned one." >&2; exit 1; }
echo "   archive sha256: OK"

tar xzf "$WORK/plugin.tar.gz" -C "$WORK"
SRC="$WORK/vllm-gguf-plugin-$PIN"
[ -d "$SRC" ] || { echo "ERROR: the archive did not unpack to $SRC" >&2; exit 1; }

say "before the series"
( cd "$SRC" && printf '%s\n' "$BEFORE" | sha256sum -c - >/dev/null ) \
  || { echo "ERROR: upstream's files are not in the state this series was cut for." >&2; exit 1; }
echo "   6 files: OK"

say "the series, ./series order"
for name in "${SERIES[@]}"; do
  # The Gluon modules are files rather than a patch, and the dispatch patch
  # carries their import, so they have to be in place before it applies.
  if [ "$name" = "gguf-gluon-dispatch.patch" ]; then
    mkdir -p "$SRC/vllm_gguf_plugin/triton/gluon"
    cp "$HERE"/gluon/*.py "$SRC/vllm_gguf_plugin/triton/gluon/"
    echo "   gluon/ -> vllm_gguf_plugin/triton/gluon/ ($(ls "$HERE"/gluon/*.py | wc -l) modules)"
  fi
  patch -p1 --dry-run --batch --fuzz 0 -d "$SRC" < "$HERE/patches/$name" >/dev/null || {
    echo "FAILED: $name does not apply to the pinned source." >&2; exit 1; }
  patch -p1 --batch --fuzz 0 -d "$SRC" < "$HERE/patches/$name" >/dev/null
  echo "   $name"
done

say "after the series"
( cd "$SRC" && printf '%s\n' "$AFTER" | sha256sum -c - >/dev/null ) \
  || { echo "ERROR: the patched files are not what this series produces." >&2; exit 1; }
# A restatement of one of the hashes above, kept because it names what the
# change was for rather than only that the bytes match.
#
# There is deliberately no companion check on the IQ3_S grid, although that is
# the other thing the series corrects. 0x3e is a byte inside the packed grid
# words and occurs in the corrected table too, so counting it asserts nothing;
# the hash of ggml-common.h above is what actually pins the corrected table.
grep -q "launch_mul_mat_vec_q_batched" "$SRC/vllm_gguf_plugin/csrc/gguf/mmvq.cuh"
echo "   6 files: OK"

say "building for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
( cd "$SRC" && TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
    "$PY" -m pip install --no-cache-dir --no-deps --no-build-isolation . )

say "what landed"
ROOT=$("$PY" -c 'import vllm_gguf_plugin, os; print(os.path.dirname(os.path.dirname(vllm_gguf_plugin.__file__)))' 2>/dev/null | tail -n1)
[ -n "$ROOT" ] && [ -d "$ROOT" ] || { echo "ERROR: cannot import vllm_gguf_plugin with $PY" >&2; exit 1; }
( cd "$ROOT" && printf '%s\n' "$AFTER" | grep -vE 'csrc/' | sha256sum -c - >/dev/null ) \
  || { echo "ERROR: the installed package does not match what was built." >&2; exit 1; }
echo "   3 files: OK"

echo "gguf-plugin: installed"
echo "run the CPU gate with:  $PY $HERE/test_gguf_rco_cpu.py"
