#!/usr/bin/env python3
"""Recompute every source drift-guard pin in fp8/.

The four Python steps under fp8/ are not patch files: they rewrite installed
sources by exact anchor matching and guard themselves with sha256 pins of the
files they read. The anchors survive a vLLM version bump or a change in
patches/series; every pin of a file that moved has to be recomputed -- in
dependency order, because the later steps pin the earlier ones' output.

    python scripts/repin-fp8-installers.py <fp8 dir> <vllm package dir>

<vllm package dir> is the patched vllm package as it stands after patches/series
and kvarn/install.sh and BEFORE the first fp8 step. Point <fp8 dir> at a copy if
you want to inspect the result before keeping it; this writes in place.

Order, and why:

  1. fp8-causal r1 reads four files straight from the tree
  2. r2 reads r1's output, so its pins depend on (1)
  3. decode_z and flat read r2's output plus the two files r2 leaves alone
  4. install_composite pins the composite's own generated unified
  5. install_composite pins its four dependency files by hash, last, because
     steps 3 and 4 rewrite two of them

Whatever a pin's new value turns out to be, it is printed. A pin that moves is a
file something upstream of it changed -- look at why before trusting the build.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import sys
from pathlib import Path


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf8") if isinstance(data, str) else data).hexdigest()


def load(path: Path, name: str, extra_paths: list[Path] = []):
    for p in extra_paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHANGES: list[tuple[str, str, str, str]] = []


def repin(path: Path, pattern: str, value: str, label: str) -> None:
    """Replace the single 64-hex group-2 match of `pattern` in `path`."""
    text = path.read_bytes().decode("utf8")
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise SystemExit(f"{path.name}: {len(matches)} matches for {label}, expected 1")
    old = matches[0].group(2)
    if old != value:
        CHANGES.append((path.name, label, old, value))
    text = text[:matches[0].start(2)] + value + text[matches[0].end(2):]
    path.write_bytes(text.encode("utf8"))


def const(name: str) -> str:
    return r"(\b" + re.escape(name) + r"\s*=\s*['\"])([0-9a-f]{64})(['\"])"


def key(name: str) -> str:
    return r"(\b" + re.escape(name) + r"\s*=\s*')([0-9a-f]{64})(')"


def entry(name: str) -> str:
    return r"('" + re.escape(name) + r"'\s*:\s*')([0-9a-f]{64})(')"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    fp8, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    causal = fp8 / "fp8-causal"
    composite = fp8 / "fp8-composite"

    # ---------------------------------------------------------------- step 1
    fp8_causal = causal / "fp8_causal.py"
    r1 = load(causal / "install_fp8_causal.py", "install_fp8_causal", [causal])
    tree = {n: (root / p).read_bytes() for n, p in r1.FILES.items()}
    for name, constant in (("unified", "PARENT_SHA"), ("helpers", "HELPER_SHA"),
                           ("backend", "BACKEND_SHA"), ("env", "ENV_SHA")):
        repin(fp8_causal, const(constant), sha(tree[name]), f"{constant} ({r1.FILES[name]})")

    # ---------------------------------------------------------------- step 2
    gen = load(fp8_causal, "fp8_causal", [causal])          # reload with new pins
    r1_unified = gen.generated_source(tree["unified"].decode("utf8")).encode("utf8")
    r1_env = gen.generated_env(tree["env"].decode("utf8")).encode("utf8")
    repin(causal / "r2/fp8_causal_r2.py", const("R1_SOURCE_SHA"), sha(r1_unified),
          "R1_SOURCE_SHA (r1 output)")
    repin(causal / "r2/install_fp8_causal_r2.py", const("ENV_INSTALLED_SHA"), sha(r1_env),
          "ENV_INSTALLED_SHA (r1 output)")

    # ---------------------------------------------------------------- step 3
    r2 = load(causal / "r2/fp8_causal_r2.py", "fp8_causal_r2", [causal, causal / "r2"])
    after_r2 = dict(tree, unified=r2.generated_source(r1_unified.decode("utf8")).encode("utf8"),
                    env=r1_env)

    segments_installer = composite / "decode_z/install_decode_segments.py"
    segments = load(segments_installer, "install_decode_segments", [composite / "decode_z"])
    for name in segments.FILES:
        repin(segments_installer, key(name), sha(after_r2[name]),
              f"PINS[{name}] ({segments.FILES[name]})")

    flat_source = composite / "flat/fp8_prefill_flat.py"
    repin(flat_source, const("PARENT_SHA"), sha(after_r2["unified"]), "PARENT_SHA (unified)")
    repin(flat_source, const("HELPER_SHA"), sha(after_r2["helpers"]), "HELPER_SHA (helpers)")

    # ---------------------------------------------------------------- step 4
    installer = composite / "install_composite.py"
    flat = load(flat_source, "fp8_prefill_flat", [composite / "flat"])
    repin(installer, entry("unified"),
          sha(flat.generated_source(after_r2["unified"].decode("utf8")).encode("utf8")),
          "EXPECTED_COMPONENTS[unified]")

    # ---------------------------------------------------------------- step 5
    # Read DEPENDENCIES without importing: install_composite verifies these very
    # hashes at import time and they are exactly what is about to be rewritten.
    module = ast.parse(installer.read_bytes().decode("utf8"))
    node = next(n.value for n in module.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "DEPENDENCIES" for t in n.targets))
    for relative in ast.literal_eval(node):
        repin(installer, entry(relative), sha((composite / relative).read_bytes()),
              f"DEPENDENCIES[{relative}]")

    print(f"{len(CHANGES)} pins moved\n")
    for filename, label, old, new in CHANGES:
        print(f"  {filename:<30} {label}\n      {old}\n   -> {new}")


if __name__ == "__main__":
    main()
