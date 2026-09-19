#!/usr/bin/env bash
set -euo pipefail

# Fail if a tracked file names a path that does not exist in this repository.
#
# A patch, a test or a note that was produced in another working tree carries
# that tree's layout with it: `git format-patch` writes a `From:`/`Source:`
# header, the prose above a hunk cites the directory the work happened in, and
# the file name itself can keep a numbering scheme that means nothing here. None
# of it resolves for anyone who clones this repo, so it is not documentation --
# it is a dangling reference, and it ages into a lie about where the code lives.
#
# The series is regenerated from commits (scripts/export-patch.sh), so the fix
# is always to correct the commit and re-export, never to edit a patch file.
#
#   bash scripts/check-no-foreign-paths.sh
#
# Runs over tracked files only, so it says the same thing here and in CI.

cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="scripts/$(basename -- "${BASH_SOURCE[0]}")"

# Each entry is <regex>\t<what it means>. Patterns are extended REs.
#
# `issues/` is deliberately NOT matched on its own: this repo's docs link
# upstream issues constantly (issues/62, issues/105). Only the zero-padded form
# is matched, which GitHub never produces and a local numbering scheme does.
RULES=(
$'(^|[^A-Za-z0-9_-])ctx/\ta context directory outside this repository'
$'(^|[^A-Za-z0-9_-])experiments/\tan experiment tree outside this repository'
$'issues/0[0-9]{3}\ta zero-padded issue number; GitHub issue links are not padded'
$'patch-series@local\ta placeholder author from a local format-patch run'
$'^Source:[[:space:]]\ta Source: header naming the tree a patch came from'
$'v0[0-9]{2}-[a-z0-9]+-20[0-9]{6}\ta dated build tag from another repository'
$'(^|[^A-Za-z0-9_-])migrations/\ta migration tree outside this repository'
$'ctx-overlay\ta build-context overlay outside this repository'
$'assemble-context\.py\ta build-context assembler outside this repository'
$'(^|[^A-Za-z0-9_-])\.dev/\ta private working directory'
$'(^|[^A-Za-z0-9_-])provenance/\ta provenance tree outside this repository'
$'(^|[^A-Za-z0-9_-])model-pipeline/\ta model pipeline outside this repository'
$'[Dd]onor\ta reference to the tree some of this work came from'
)

fail=0

# -- pass 1: file contents ---------------------------------------------------
# -I skips binary files. The check script names every pattern it looks for, so
# it would match itself on every line; it is the one file excluded.
mapfile -t FILES < <(git ls-files | grep -vxF "$SELF")
for rule in "${RULES[@]}"; do
  pattern=${rule%%$'\t'*}
  meaning=${rule#*$'\t'}
  hits=$(grep -InE -- "$pattern" "${FILES[@]}" 2>&1) && rc=0 || rc=$?
  if [ $rc -gt 1 ]; then
    # grep exits 2 for an error and 1 for "no match". Treating them alike would
    # skip a rule in silence, which is the one thing this check must not do.
    echo "FAIL: grep could not apply the rule for $meaning (exit $rc)"
    echo "$hits" | sed 's/^/    /'
    echo
    fail=1
  elif [ $rc -eq 0 ]; then
    echo "FAIL: $meaning"
    echo "$hits" | sed 's/^/    /'
    echo
    fail=1
  fi
done

# -- pass 2: file names ------------------------------------------------------
# patches/ is named by topic, and the apply order lives in patches/series (see
# its header). A leading serial number is another tree's ordering, and it puts
# two sources of truth in the repo.
if numbered=$(printf '%s\n' "${FILES[@]}" | grep -E '/[0-9]{4}[a-z]?-'); then
  echo "FAIL: a file name carrying another tree's serial number"
  echo "$numbered" | sed 's/^/    /'
  echo
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "foreign paths: FAILED -- fix the commit these were exported from, then re-export." >&2
  exit 1
fi
echo "foreign paths: OK ($(printf '%s\n' "${FILES[@]}" | wc -l) tracked files, ${#RULES[@]} patterns)"
