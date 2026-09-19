"""Host-only target Z16/32 integration against an exact parent. No kernel source changes."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

# PARENT: The tree after fp8-causal r2, which is what PINS below is of. A docker
# image id stood here; on this tree the parent is not an image, it is four files,
# and their sha256 is the whole of the claim.
FLAG='VLLM_TRITON_FP8_MQ3D_SEGMENTS'
FILES=dict(unified='v1/attention/ops/triton_unified_attention.py',
           helpers='v1/attention/ops/triton_attention_helpers.py',
           backend='v1/attention/backends/triton_attn.py',env='envs.py')
PINS=dict(unified='b428d0a7680242917e825e5146bb7b35e3a013eb984f3fc34c923d33eebe65c7',
          helpers='8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95',
          backend='c01afe31a641ce6c772b41b6b319d6d5a64810a3fbdf6edc727d006e83a4bdde',
          env='6761a702a2f5612f4e5615ba7a57587a46a30cf64a769bc36959fb09945fa978')
ENV_ENTRY=f'    "{FLAG}": lambda: int(os.environ.get("{FLAG}", "16")),\n'
ANCHOR='NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments\n'
HELPER='''
# Startup-frozen, target-only experiment. Existing/default geometry is Z16.
_FP8_MQ3D_SEGMENTS = envs.VLLM_TRITON_FP8_MQ3D_SEGMENTS
if _FP8_MQ3D_SEGMENTS not in (16, 32):
    raise ValueError("VLLM_TRITON_FP8_MQ3D_SEGMENTS must be 16 or 32")


def _select_fp8_mq3d_segments(num_heads_q, num_heads_kv, headdim, block_size, kv_quant_mode):
    if (
        _FP8_MQ3D_SEGMENTS == 32
        and envs.VLLM_TRITON_FP8_MQ3D
        and kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
        and num_heads_q == 24 and num_heads_kv == 4 and headdim == 256
        and block_size == 880
        and current_platform.is_device_capability(89)
    ):
        return 32
    return NUM_PAR_SOFTMAX_SEGMENTS
'''
OLD_SELECT='        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS\n'

# ---- THE POOL HAS TO KNOW THE COUNT -----------------------------------------
# vLLM 0.29 moved the three 3D-softmax scratch buffers into a pool that
# mq3d_scratch_plan() sizes, and that plan sizes rows with the module constant.
# A builder that selects 32 segments against a pool built for 16 writes past the
# end of all three buffers. The pool key already carries a slot for the segment
# count -- upstream's own comment over _MQ3D_SCRATCH_POOL names it -- so the
# count becomes a field of the plan and every user reads the plan instead of the
# constant. With the flag at its default the selected count IS the constant, so
# the pool, its key and the byte count are exactly what they were.
#
# The attention Impl keeps calling the plan with the default, inside the memory
# profile. When the flag is on the builder's key therefore differs from the
# Impl's, the builder allocates its own set after the profile, and upstream's
# own WARNING says so with the byte count. That is the cost of Z32, in the boot
# record, where it can be read back.
SCRATCH = (
    ('    max_num_seqs: int\n',
     '    max_num_seqs: int\n'
     '    segments: int                  # PARALLEL SOFTMAX SEGMENTS  (allocation)\n'),
    ('def mq3d_scratch_plan(vllm_config: VllmConfig, num_heads_q: int) -> Mq3dScratchPlan:\n',
     'def mq3d_scratch_plan(\n'
     '    vllm_config: VllmConfig, num_heads_q: int, segments: int = NUM_PAR_SOFTMAX_SEGMENTS\n'
     ') -> Mq3dScratchPlan:\n'),
    ('    row_bytes = 4 * (\n'
     '        num_heads_q * NUM_PAR_SOFTMAX_SEGMENTS * headdim_padded\n'
     '        + 2 * num_heads_q * NUM_PAR_SOFTMAX_SEGMENTS\n'
     '    )\n',
     '    row_bytes = 4 * (\n'
     '        num_heads_q * segments * headdim_padded\n'
     '        + 2 * num_heads_q * segments\n'
     '    )\n'),
    ('    return Mq3dScratchPlan(\n        num_heads_q=num_heads_q,\n',
     '    return Mq3dScratchPlan(\n        segments=segments,\n        num_heads_q=num_heads_q,\n'),
    ('            (rows, plan.num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS, plan.headdim_padded),\n',
     '            (rows, plan.num_heads_q, plan.segments, plan.headdim_padded),\n'),
    ('            (rows, plan.num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS),\n',
     '            (rows, plan.num_heads_q, plan.segments),\n'),
    ('    key = (dev.type, dev.index, plan.num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS, plan.headdim_padded)\n',
     '    key = (dev.type, dev.index, plan.num_heads_q, plan.segments, plan.headdim_padded)\n'),
    ('        plan = mq3d_scratch_plan(self.vllm_config, self.num_heads_q)\n',
     '        plan = mq3d_scratch_plan(\n'
     '            self.vllm_config, self.num_heads_q, self.num_par_softmax_segments\n'
     '        )\n'),
)


def once(source, old, new):
    assert source.count(old) == 1, ('anchor is not unique', old[:60])
    return source.replace(old, new, 1)

NEW_SELECT='''        self.num_par_softmax_segments = _select_fp8_mq3d_segments(
            self.num_heads_q, self.num_heads_kv, self.headdim,
            self.block_size, kv_cache_spec.kv_quant_mode,
        )
'''


def sha(value):
    return hashlib.sha256(value.encode('utf8') if isinstance(value,str) else value).hexdigest()


def reversed_backend(output):
    """Undo every replacement generated_backend() made, in reverse order."""
    restored=output
    for old,new in reversed(SCRATCH):restored=once(restored,new,old)
    return once(once(restored,NEW_SELECT,OLD_SELECT),ANCHOR+HELPER,ANCHOR)


def generated_backend(parent):
    assert sha(parent)==PINS['backend'],'Unknown backend parent'
    assert parent.count(ANCHOR)==parent.count(OLD_SELECT)==1
    result=once(once(parent,ANCHOR,ANCHOR+HELPER),OLD_SELECT,NEW_SELECT)
    for old,new in SCRATCH:result=once(result,old,new)
    assert reversed_backend(result)==parent,'Backend does not reverse to its parent'
    compile(result,'<target-decode-segments-backend>','exec')
    return result


def generated_env(parent):
    assert sha(parent)==PINS['env'],'Unknown env parent'
    anchor='    "VLLM_TRITON_FP8_MQ3D": lambda:'
    assert parent.count(anchor)==1 and FLAG not in parent
    result=parent.replace(anchor,ENV_ENTRY+anchor,1)
    assert result.replace(ENV_ENTRY,'',1)==parent
    compile(result,'<target-decode-segments-env>','exec')
    return result


def allocation_nodes(source):
    """Exact installed constructor selection/capacity/three-allocation statements."""
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TritonAttentionMetadataBuilder')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    def assigns(n,name):
        return isinstance(n,ast.Assign) and any(ast.unparse(t)==f'self.{name}' for t in n.targets)
    start=next(i for i,n in enumerate(init.body) if assigns(n,'num_par_softmax_segments'))
    end=next(i for i,n in enumerate(init.body) if assigns(n,'softmax_segm_expsum'))
    return copy.deepcopy(init.body[start:end+1])


def pool_nodes(source):
    """The four module-level definitions the scratch pool is made of, by name."""
    tree=ast.parse(source)
    wanted=('Mq3dScratchPlan','mq3d_scratch_plan','Mq3dScratch','mq3d_scratch_acquire')
    found={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted}
    assert set(found)==set(wanted),sorted(set(wanted)-set(found))
    return copy.deepcopy([found[n] for n in wanted])


def prepare(root,causal_archive):
    original={n:(root/path).read_bytes() for n,path in FILES.items()}
    assert {n:sha(b) for n,b in original.items()}==PINS,'Exact b80 source seal mismatch'
    prior_bytes=(causal_archive/'manifest.json').read_bytes()
    prior=json.loads(prior_bytes)
    assert prior['files']==FILES and prior['installed']==PINS,'Prior four-file manifest mismatch'
    assert prior['delivery_revision']=='r2-prefill-only'
    assert prior['flag']=='VLLM_TRITON_FP8_CAUSAL_FULL' and prior['default']=='0' and prior['frozen'] is True
    updated=dict(original)
    updated['backend']=generated_backend(original['backend'].decode('utf8')).encode('utf8')
    updated['env']=generated_env(original['env'].decode('utf8')).encode('utf8')
    assert updated['unified']==original['unified'] and updated['helpers']==original['helpers']
    declaration=dict(flag=FLAG,default=16,supported=[16,32],frozen=True,
        scope='SM89 FP8-per-tensor MQ3D target H24/Hkv4/D256/B880 only; every other builder Z16',
        kernel_and_helpers_unchanged=True,packed_scratch_allocated_before_capture=True)
    continued=copy.deepcopy(prior)
    continued['installed']={n:sha(b) for n,b in updated.items()}
    continued['decode_segments']=declaration
    manifest=dict(files=FILES,parent=PINS,installed=continued['installed'],
        **declaration,installer_sha256=sha(Path(__file__).read_bytes()),
        causal_parent_manifest_sha256=sha(prior_bytes),causal_continued_manifest=continued)
    return original,updated,prior_bytes,continued,manifest


def validate_installed(root,archive,causal_archive):
    manifest=json.loads((archive/'manifest.json').read_text())
    assert manifest['parent']==PINS and manifest['files']==FILES
    assert manifest['installer_sha256']==sha(Path(__file__).read_bytes())
    assert manifest['flag']==FLAG and manifest['default']==16 and manifest['supported']==[16,32]
    assert manifest['frozen'] is True and manifest['kernel_and_helpers_unchanged'] is True
    original={n:(archive/'parent'/f'{n}.py').read_bytes() for n in FILES}
    assert {n:sha(b) for n,b in original.items()}==PINS
    expected=dict(PINS,backend=sha(generated_backend(original['backend'].decode())),
                  env=sha(generated_env(original['env'].decode())))
    assert manifest['installed']==expected
    assert {n:sha((root/p).read_bytes()) for n,p in FILES.items()}==expected
    prior=(archive/'parent/causal-manifest.json').read_bytes()
    assert sha(prior)==manifest['causal_parent_manifest_sha256']
    assert json.loads((causal_archive/'manifest.json').read_text())==manifest['causal_continued_manifest']
    assert manifest['causal_continued_manifest']['installed']==expected
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vllm-root',type=Path,default=Path('/usr/local/lib/python3.12/dist-packages/vllm'))
    p.add_argument('--causal-archive',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--verify-installed',action='store_true')
    a=p.parse_args()
    if a.verify_installed:
        print(json.dumps(validate_installed(a.vllm_root,a.archive,a.causal_archive),indent=2));return
    original,updated,prior,continued,manifest=prepare(a.vllm_root,a.causal_archive)
    if not a.check_only:
        assert not (a.archive/'manifest.json').exists() and not (a.archive/'parent').exists(),'Preserve prior archive'
        (a.archive/'parent').mkdir(parents=True)
        for n,data in original.items():(a.archive/'parent'/f'{n}.py').write_bytes(data)
        (a.archive/'parent/causal-manifest.json').write_bytes(prior)
        for n in ('env','backend'):(a.vllm_root/FILES[n]).write_bytes(updated[n])
        (a.causal_archive/'manifest.json').write_text(json.dumps(continued,indent=2)+'\n')
        (a.archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        validate_installed(a.vllm_root,a.archive,a.causal_archive)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
