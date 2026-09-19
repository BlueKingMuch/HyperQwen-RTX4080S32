"""Build-time atomic seal for the flat prefill mapping and the target Z32 selector.

Two independent generators, one manifest. `flat/fp8_prefill_flat.py` rewrites the
Triton kernel's prefill index mapping; `decode_z/install_decode_segments.py`
rewrites the backend's softmax-segment selector. Neither is a patch file: both
match exact anchors and guard themselves with sha256 pins of what they read, and
both prove their output reverses byte-for-byte back to the parent.

A third feature -- the sparse Mamba-align block reclaim, behind
VLLM_MAMBA_ALIGN_SPARSE_RECLAIM -- is not installed here. This tree already
retires those blocks: patches/mamba-align-retire-null-gaps.patch defines
SingleTypeKVCacheManager._remove_blocks_in_range for mamba_cache_mode == "align"
with no flag to turn it on. Installing a second definition would shadow it.

    python install_composite.py --vllm-root <vllm package> --causal-archive <dir>

The causal archive is the one fp8-causal/r2 wrote: this step reads its manifest
and carries it forward, so the chain from the unpatched tree to here stays a
single unbroken record.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DEPENDENCIES = {
    'flat/fp8_prefill_flat.py': 'd0d6ba88ad303929a4bd749338082872214b38963bf5c47c9fa81a854f0c3843',
    'flat/test_fp8_prefill_flat_cpu.py': '8b6c3008f03f44943c747d78367c7a3407c087394933f0b5618284281df623b2',
    'decode_z/install_decode_segments.py': '143d80c863a8b11eeb14a978866c0a26f3fc65f1ea6444ae3c8beed846660f8a',
    'decode_z/test_decode_segments_cpu.py': 'd89331aa23128821b2a30fa656f6b535138bdbf0636459f4ee9455783f965992',
}


def sha(value):
    return hashlib.sha256(value.encode('utf8') if isinstance(value, str) else value).hexdigest()


def load_dependency(relative, module_name):
    path = HERE / relative
    assert sha(path.read_bytes()) == DEPENDENCIES[relative], ('Dependency changed', relative)
    if module_name in sys.modules:
        module = sys.modules[module_name]
        assert Path(module.__file__).resolve() == path.resolve(), ('Unexpected import', module_name)
        assert sha(Path(module.__file__).read_bytes()) == DEPENDENCIES[relative]
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


flat = load_dependency('flat/fp8_prefill_flat.py', 'fp8_prefill_flat')
segments = load_dependency('decode_z/install_decode_segments.py', 'install_decode_segments')
FILES, PINS = dict(segments.FILES), dict(segments.PINS)
assert len(FILES) == 4 and set(FILES) == set(PINS)
# The two generators agree on their parent because they pin the same bytes of it.
assert PINS['unified'] == flat.PARENT_SHA and PINS['helpers'] == flat.HELPER_SHA
FLAT_ENV_ENTRY = ('    "VLLM_TRITON_FP8_PREFILL_FLAT": lambda: {"0": 0, "1": 1}'
                  '[os.environ.get("VLLM_TRITON_FP8_PREFILL_FLAT", "0")],\n')
EXPECTED_COMPONENTS = {
    'unified': '60778cbd528eae576bea60abec19f22441ed38e958b8a4d98b9fd667f7cfc955',
}


def generated_env(parent):
    # Run the unmodified independent generator against its exact parent, then
    # compose its proved one-entry delta with flat's. Never relax a parent pin.
    z = segments.generated_env(parent)
    assert z.replace(segments.ENV_ENTRY, '', 1) == parent
    anchor = '    "VLLM_TRITON_FP8_MQ3D": lambda:'
    assert parent.count(anchor) == 1 and flat.FLAG not in parent
    entries = FLAT_ENV_ENTRY + segments.ENV_ENTRY
    output = parent.replace(anchor, entries + anchor, 1)
    assert output.replace(entries, '', 1) == parent
    tree = ast.parse(output)
    factors = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compile_factors')
    ignored = next(n.value for n in factors.body if isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name) and n.target.id == 'ignored_factors')
    assert not {flat.FLAG, segments.FLAG} & ast.literal_eval(ignored)
    compile(tree, '<composite-frozen-registry>', 'exec')
    return output


def derive(original, prior_bytes):
    assert set(original) == set(FILES), 'Exact four-file parent required'
    assert {n: sha(b) for n, b in original.items()} == PINS, 'Parent source drift'
    assert {p: sha((HERE / p).read_bytes()) for p in DEPENDENCIES} == DEPENDENCIES
    prior = json.loads(prior_bytes)
    assert prior['files'] == FILES and prior['installed'] == PINS, 'Prior four-file manifest mismatch'
    assert prior['delivery_revision'] == 'r2-prefill-only'
    assert prior['flag'] == 'VLLM_TRITON_FP8_CAUSAL_FULL' and prior['default'] == '0'
    assert prior['frozen'] is True
    updated = dict(original)
    updated['unified'] = flat.generated_source(original['unified'].decode('utf8')).encode('utf8')
    updated['backend'] = segments.generated_backend(original['backend'].decode('utf8')).encode('utf8')
    updated['env'] = generated_env(original['env'].decode('utf8')).encode('utf8')
    assert {n for n in FILES if updated[n] != original[n]} == {'unified', 'backend', 'env'}
    assert {n: sha(updated[n]) for n in EXPECTED_COMPONENTS} == EXPECTED_COMPONENTS
    # Reverse-byte proofs are independent of selected runtime flags.
    flat_proof = flat.assert_identity(original['unified'].decode(), updated['unified'].decode())
    assert segments.reversed_backend(updated['backend'].decode()) == original['backend'].decode()
    for n, data in updated.items():
        compile(data.decode('utf8'), '<composite-' + n + '>', 'exec')
    features = {
        'flat': dict(flag=flat.FLAG, default=0, supported=[0, 1], frozen=True,
                     compile_factors_included=True, new_registry_entry=True,
                     scope='Exact original native2D flat mapping; SM89/B880/H24-Hkv4/D256; full-causal required'),
        'decode_segments': dict(flag=segments.FLAG, default=16, supported=[16, 32], frozen=True,
                     compile_factors_included=True, new_registry_entry=True,
                     scope='Exact target-only selector; MQ3D/SM89/FP8/H24-Hkv4/D256/B880; other builders Z16',
                     scratch_pool='Mq3dScratchPlan carries the selected segment count; at the '
                                  'default it is NUM_PAR_SOFTMAX_SEGMENTS and the pool, its key '
                                  'and its byte count are unchanged'),
    }
    manifest = dict(schema=1, kind='fp8-composite', revision='r1',
        files=FILES, parent=PINS, installed={n: sha(b) for n, b in updated.items()},
        installer_sha256=sha(Path(__file__).read_bytes()), dependency_sha256=DEPENDENCIES,
        features=features, flat_identity=flat_proof,
        not_installed_here={'mamba_reclaim': 'VLLM_MAMBA_ALIGN_SPARSE_RECLAIM; '
            'patches/mamba-align-retire-null-gaps.patch already defines '
            '_remove_blocks_in_range for mamba_cache_mode == "align", unconditionally'},
        parent_causal_manifest_sha256=sha(prior_bytes), parent_causal_manifest=prior,
        archive_policy='No inherited archive writes; the four parent source files and the prior manifest are retained here',
        provenance=dict(flat_generated_sha256=sha(updated['unified']),
            z_backend_sha256=sha(updated['backend'])),
        unchanged_files=sorted(set(FILES) - {'unified', 'backend', 'env'}),
        release_activation='External explicit ENV only; image defaults Flat0/Z16; no activation by installer',
        GPU_or_model_composition_gate_claim=False)
    return updated, manifest


def prepare(root, causal_archive):
    original = {n: (root / path).read_bytes() for n, path in FILES.items()}
    prior = (causal_archive / 'manifest.json').read_bytes()
    return original, prior, *derive(original, prior)


def verify_seal(archive, causal_archive):
    """The seal on its own: re-derive both generated files from the archived
    parents and compare the whole manifest. Reads nothing from the live tree,
    because a later step legitimately rewrites two of these four files --
    fp8-paged does, and fp8-paged/PINS is what says the tree is still right."""
    manifest = json.loads((archive / 'manifest.json').read_bytes())
    original = {n: (archive / 'parent' / f'{n}.py').read_bytes() for n in FILES}
    prior = (archive / 'parent/causal-manifest.json').read_bytes()
    _, expected = derive(original, prior)
    assert manifest == expected, 'Composite whole-manifest seal mismatch'
    assert (causal_archive / 'manifest.json').read_bytes() == prior, 'Inherited parent manifest was changed'
    return manifest


def validate_installed(root, archive, causal_archive):
    manifest = verify_seal(archive, causal_archive)
    assert {n: sha((root / path).read_bytes()) for n, path in FILES.items()} == manifest['installed']
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vllm-root', type=Path)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--causal-archive', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-installed', action='store_true')
    mode.add_argument('--verify-seal', action='store_true')
    a = p.parse_args()
    if a.verify_seal:
        print(json.dumps(verify_seal(a.archive, a.causal_archive), indent=2)); return
    if a.verify_installed:
        print(json.dumps(validate_installed(a.vllm_root, a.archive, a.causal_archive), indent=2)); return
    assert a.vllm_root is not None, '--vllm-root is required unless --verify-seal'
    original, prior, updated, manifest = prepare(a.vllm_root, a.causal_archive)
    if not a.check_only:
        assert not a.archive.exists(), 'Never overwrite prior composite state'
        (a.archive / 'parent').mkdir(parents=True)
        for n, data in original.items(): (a.archive / 'parent' / f'{n}.py').write_bytes(data)
        (a.archive / 'parent/causal-manifest.json').write_bytes(prior)
        # Build-time only: no serving process is allowed during this transaction.
        # All three generated files are ready/verified before any live-file write.
        # Completion seal is published LAST, never over a partial installation.
        for n in ('env', 'unified', 'backend'):
            (a.vllm_root / FILES[n]).write_bytes(updated[n])
        assert {n: sha((a.vllm_root / path).read_bytes()) for n, path in FILES.items()} == manifest['installed']
        assert (a.causal_archive / 'manifest.json').read_bytes() == prior
        (a.archive / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
        validate_installed(a.vllm_root, a.archive, a.causal_archive)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
