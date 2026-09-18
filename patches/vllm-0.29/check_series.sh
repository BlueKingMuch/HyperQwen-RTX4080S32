#!/usr/bin/env bash
set -euo pipefail

# Validate the 0.29.0 series against a pristine checkout of vLLM v0.29.0.
#
# The sibling of patches/check_vllm_series.sh, for the series in this directory.
# It is a separate file rather than a flag on that one because the two series
# answer to different pins and different histories, and the 0.28.0 checker is
# the one the build and the CI job depend on: a shared script would make a
# change here able to break that.
#
# Two passes, because two tools are in play and they answer different questions:
#
#   1. Every patch, in the order of ./series, applied with GNU `patch` -- the
#      tool that installs this stack. This is the pass that says "this series
#      still applies to the pin it claims".
#   2. The same series again with `git apply`, which is strict about offsets and
#      rejects a hand-edited hunk header immediately.
#
# The 0.28.0 checker can only afford pass 2 on five patches: the rest have
# drifted far enough from the tree they were cut against that GNU patch accepts
# them and git apply does not. Every patch here is exported from its own commit
# on a branch rooted at v0.29.0, so all eighteen still carry exact metadata and
# all eighteen are checked. When that stops being true the fix is to regenerate
# the patch from its commit (scripts/export-patch.sh), never to edit the file or
# to move it out of pass 2.
#
#   git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git /tmp/vllm
#   bash patches/vllm-0.29/check_series.sh /tmp/vllm/vllm
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SOURCE=${1:?usage: bash patches/vllm-0.29/check_series.sh /path/to/vllm-v0.29.0/vllm  (the package directory inside the checkout, not its root)}
VLLM_SOURCE=$(cd -- "$VLLM_SOURCE" && pwd)

git -C "$VLLM_SOURCE" rev-parse --is-inside-work-tree >/dev/null

# The patches address files relative to the installed vllm package, not to the
# checkout root. `git apply` resolves a patch path against the REPOSITORY root
# and silently skips ("Skipped patch '...'") anything outside the subdirectory
# it runs in, so it must run from the repository root with the prefix named.
#
# The prefix comes from `rev-parse --show-prefix` rather than from subtracting
# `--show-toplevel` out of the directory: on Windows those two are formatted
# differently (`C:/src/vllm` against `/c/src/vllm`), the subtraction silently
# matches nothing, and the fallback leaves the prefix at `.` -- which points
# git apply at the checkout root, where none of these paths exist.
GIT_ROOT=$(cd -- "$(git -C "$VLLM_SOURCE" rev-parse --show-toplevel)" && pwd)
PREFIX=$(git -C "$VLLM_SOURCE" rev-parse --show-prefix)
PREFIX=${PREFIX%/}
[ -n "$PREFIX" ] || PREFIX=.

# ./series is the single source of truth for the apply order. It and the
# directory must agree exactly: a patch absent from series is never applied,
# and one listed but missing is a typo.
SERIES=()
while IFS= read -r name; do
  SERIES+=("$name")
done < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series")
ON_DISK=$(for f in "$HERE"/*.patch; do basename "$f"; done | sort)
IN_SERIES=$(printf '%s\n' "${SERIES[@]}" | sort)
[ "$ON_DISK" = "$IN_SERIES" ] || {
  echo "ERROR: ./series and this directory disagree:" >&2
  comm -3 <(printf '%s\n' "$ON_DISK") <(printf '%s\n' "$IN_SERIES") | sed 's/^/    /' >&2
  exit 1
}

echo "== pass 1: the whole series, GNU patch, ./series order"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
# --fuzz 0: an offset means the context matched exactly and the file merely grew around it; fuzz means the
# context did NOT match and GNU patch accepted an approximate anchor. The first is benign and reported, the
# second is a patch cut against a tree that no longer exists, and it fails here instead of landing by guess.
count=0; offset=0
for name in "${SERIES[@]}"; do
  out=$(patch -p1 --forward --no-backup-if-mismatch --fuzz 0 -d "$VLLM_SOURCE" < "$HERE/$name" 2>&1) || {
    echo "FAILED: $name (a hunk's context does not exist in this tree; regenerate the patch from its commit)"
    echo "$out" | sed 's/^/    /'; exit 1
  }
  n=$(printf '%s\n' "$out" | grep -c "offset" || true)
  [ "$n" -gt 0 ] && { echo "   $name (applied, $n hunk(s) with an offset, context exact)"; offset=$((offset+1)); }
  count=$((count+1))
done
if git -C "$GIT_ROOT" diff --quiet; then
  echo "ERROR: the series applied but changed nothing -- the paths did not resolve." >&2
  exit 1
fi
echo "   $count patches applied with exact context, $offset of them at an offset, 0 with fuzz"

echo "== pass 2: the whole series again, git apply --check"
git -C "$GIT_ROOT" checkout -q -- . && git -C "$GIT_ROOT" clean -qfd
for name in "${SERIES[@]}"; do
  git -C "$GIT_ROOT" apply --check --whitespace=error -p1 --directory="$PREFIX" < "$HERE/$name"
  git -C "$GIT_ROOT" apply --whitespace=error -p1 --directory="$PREFIX" < "$HERE/$name"
done
echo "   ${#SERIES[@]} patches accepted with exact hunk metadata"
if git -C "$GIT_ROOT" diff --quiet; then
  echo "ERROR: pass 2 applied but changed nothing." >&2
  exit 1
fi

git -C "$GIT_ROOT" diff --check
echo "0.29.0 series integrity: OK"
