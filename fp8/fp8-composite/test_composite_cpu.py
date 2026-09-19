"""The two independent proofs, unchanged, plus the composite all-OFF/seal tests.

Stdlib only -- no Torch, no CUDA, no model. Run it two ways:

    python test_composite_cpu.py --vllm-root <vllm package>   before installing
    python test_composite_cpu.py --installed --vllm-root <..> after installing

The first reads the parent files straight from the tree and derives what the
installer would write; the second reads the archived parents and additionally
checks that the live tree still matches the published seal.
"""
import argparse
import ast
import copy
import itertools
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import install_composite as gen


class MemoryPath:
    def __init__(self, data, key): self.data,self.key=data,key
    def __truediv__(self, key): return MemoryPath(self.data,self.key+'/'+str(key))
    def read_bytes(self): return self.data[self.key]


def registry_proof(source):
    tree=ast.parse(source)
    registry=next(n.value for n in tree.body if isinstance(n,ast.AnnAssign)
                  and isinstance(n.target,ast.Name) and n.target.id=='environment_variables')
    flags=(gen.flat.FLAG,gen.segments.FLAG)
    entries={k.value:v for k,v in zip(registry.keys,registry.values)
             if isinstance(k,ast.Constant) and k.value in flags}
    assert set(entries)==set(flags)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_factors')
    utils=ModuleType('vllm.config.utils'); utils.normalize_value=lambda x:x
    saved={n:sys.modules.get(n) for n in ('vllm','vllm.config','vllm.config.utils')}
    sys.modules.update({'vllm':ModuleType('vllm'),'vllm.config':ModuleType('vllm.config'),'vllm.config.utils':utils})
    hashes=set(); rejected=0
    prior_env={name:os.environ.get(name) for name in flags}
    try:
        for flat,z in itertools.product((0,1),(16,32)):
            values=dict(zip(flags,(flat,z)))
            env={name:str(value) for name,value in values.items()}
            getters={name:eval(compile(ast.Expression(node),'<registered-getter>','eval'),
                               {'os':NS(environ=env,getenv=env.get)}) for name,node in entries.items()}
            assert {name:get() for name,get in getters.items()}==values
            ns={'os':NS(environ=env,getenv=env.get),'environment_variables':getters}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual-compile-factors>','exec'),ns)
            factors=ns['compile_factors']()
            assert {name:factors[name] for name in flags}==values
            hashes.add(gen.sha(json.dumps(factors,sort_keys=True)))
            os.environ.update(env)
            f={};exec(gen.flat.FROZEN,f)
            os.environ[gen.flat.FLAG]=str(1-flat)
            assert f['_FP8_PREFILL_FLAT']==bool(flat)
            zenv=NS(VLLM_TRITON_FP8_MQ3D_SEGMENTS=z)
            zns=dict(envs=zenv);exec(gen.segments.HELPER,zns)
            zenv.VLLM_TRITON_FP8_MQ3D_SEGMENTS=48-z
            assert zns['_FP8_MQ3D_SEGMENTS']==z
        assert len(hashes)==4
        bad_values={gen.flat.FLAG:('', '2','-1','true','01','0.0'),
                    gen.segments.FLAG:('', '0','8','64','32.0','garbage')}
        for flag, raws in bad_values.items():
            for raw in raws:
                env={flag:raw}
                getter=eval(compile(ast.Expression(entries[flag]),'<bad-getter>','eval'),{'os':NS(environ=env)})
                try:
                    value=getter()
                    if flag==gen.segments.FLAG:
                        exec(gen.segments.HELPER,{'envs':NS(VLLM_TRITON_FP8_MQ3D_SEGMENTS=value)})
                except (ValueError,KeyError,TypeError): rejected+=1
                else: raise AssertionError(('Invalid flag accepted',flag,raw))
        defaults={name:eval(compile(ast.Expression(node),'<default-getter>','eval'),{'os':NS(environ={})})()
                  for name,node in entries.items()}
        assert defaults==dict(zip(flags,(0,16)))
    finally:
        for name,value in prior_env.items():
            if value is None: os.environ.pop(name,None)
            else: os.environ[name]=value
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    return dict(distinct_flag_factor_hashes=len(hashes),invalid_settings=rejected,defaults=defaults)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vllm-root',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--causal-archive',type=Path,required=True)
    p.add_argument('--installed',action='store_true')
    args=p.parse_args()
    ztest=gen.load_dependency('decode_z/test_decode_segments_cpu.py','composite_segments_cpu')
    ftest=gen.load_dependency('flat/test_fp8_prefill_flat_cpu.py','composite_flat_cpu')
    if args.installed:
        gen.validate_installed(args.vllm_root,args.archive,args.causal_archive)
        original={n:(args.archive/'parent'/f'{n}.py').read_bytes() for n in gen.FILES}
        prior=(args.archive/'parent/causal-manifest.json').read_bytes()
    else:
        original={n:(args.vllm_root/path).read_bytes() for n,path in gen.FILES.items()}
        prior=(args.causal_archive/'manifest.json').read_bytes()
    updated,manifest=gen.derive(original,prior)
    # Execute the original independent test function without changing any oracle,
    # tolerance or arithmetic; only substitute the exact parent bytes in hand.
    old_argv=sys.argv;old_z=ztest.parent_sources
    try:
        sys.argv=[__file__,'--vllm-root',str(args.vllm_root)]
        ztest.parent_sources=lambda *a,**k:{n:original[n] for n in gen.segments.FILES}
        ztest.main()
    finally:
        ztest.parent_sources=old_z;sys.argv=old_argv
    flat_counts=ftest.owner_checks(updated['unified'].decode(),updated['helpers'].decode())
    flat_counts['host_cases']=ftest.host_checks()
    flags=registry_proof(updated['env'].decode())
    # Composite code is byte-identical to each independent generated module.
    assert updated['backend']==gen.segments.generated_backend(original['backend'].decode()).encode()
    assert updated['unified']==gen.flat.generated_source(original['unified'].decode()).encode()
    entries=gen.FLAT_ENV_ENTRY+gen.segments.ENV_ENTRY
    assert updated['env'].decode().replace(entries,'',1)==original['env'].decode()
    # The third feature the seal could have carried is absent on purpose, and the
    # tree it is absent from is the one that already carries it, unconditionally.
    manager=(args.vllm_root/'v1/core/single_type_kv_cache_manager.py').read_text(encoding='utf8')
    assert 'def _remove_blocks_in_range(' in manager
    assert 'VLLM_MAMBA_ALIGN_SPARSE_RECLAIM' not in manager
    assert set(manifest['not_installed_here'])=={'mamba_reclaim'}
    files={'root/'+gen.FILES[n]:b for n,b in updated.items()}
    files.update({'archive/parent/'+n+'.py':b for n,b in original.items()})
    files.update({'archive/parent/causal-manifest.json':prior,'causal/manifest.json':prior,
                  'archive/manifest.json':json.dumps(manifest).encode()})
    def verify(data):return gen.validate_installed(MemoryPath(data,'root'),MemoryPath(data,'archive'),MemoryPath(data,'causal'))
    verify(files)
    rejected=0
    for name in gen.FILES:
        bad=dict(original);bad[name]+=b'\n'
        try:gen.derive(bad,prior)
        except AssertionError:rejected+=1
        else:raise AssertionError('Parent drift accepted')
        bad=dict(files);bad['root/'+gen.FILES[name]]+=b'\n'
        try:verify(bad)
        except AssertionError:rejected+=1
        else:raise AssertionError('Installed drift accepted')
    for field in ('files','parent','installed','features','dependency_sha256'):
        for add in (True,False):
            broken=copy.deepcopy(manifest)
            if add:broken[field]['unexpected']='invalid'
            else:broken[field].pop(next(iter(broken[field])))
            bad=dict(files);bad['archive/manifest.json']=json.dumps(broken).encode()
            try:verify(bad)
            except AssertionError:rejected+=1
            else:raise AssertionError('Malformed seal accepted')
    for key in ('causal/manifest.json','archive/parent/causal-manifest.json','archive/parent/backend.py'):
        bad=dict(files);bad[key]+=b'\n'
        try:verify(bad)
        except AssertionError:rejected+=1
        else:raise AssertionError('Historical parent drift accepted')
    print(json.dumps(dict(result='FP8_COMPOSITE_CPU_PASS',installed=manifest['installed'],
        default_all_off_equivalent=True,flat=flat_counts,
        registry=flags,rejected_seals=rejected,installed_image_checked=args.installed,GPU=False),indent=2))


if __name__=='__main__':main()
