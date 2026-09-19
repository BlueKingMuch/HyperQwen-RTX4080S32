"""Actual host selection/allocation AST and four-file seal proof; stdlib only."""
import argparse
import ast
import copy
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import install_decode_segments as gen


class FakeTorch:
    float32='float32'
    def __init__(self):self.calls=[]
    def empty(self,shape,**kw):
        value=NS(shape=tuple(shape),dtype=kw['dtype'],device=kw['device'])
        self.calls.append(value)
        return value
    def empty_like(self,other):return NS(shape=other.shape,dtype=other.dtype,device=other.device)
    @staticmethod
    def device(value):return NS(type=str(value),index=None)


class MemoryPath:
    def __init__(self,data,key):self.data,self.key=data,key
    def __truediv__(self,key):return MemoryPath(self.data,self.key+'/'+str(key))
    def read_bytes(self):return self.data[self.key]


# The four parent files are whatever the tree holds after fp8-causal r2; PINS
# below is what says whether those are the right bytes. fp8/install.sh passes
# --vllm-root, and the composite gate substitutes this function outright.
def parent_sources(installed,root=None,archive=None,causal_archive=None):
    if installed:
        gen.validate_installed(root,archive,causal_archive)
        return {n:(archive/'parent'/f'{n}.py').read_bytes() for n in gen.FILES}
    return {n:(root/path).read_bytes() for n,path in gen.FILES.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--installed',action='store_true')
    p.add_argument('--vllm-root',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    p.add_argument('--causal-archive',type=Path)
    a=p.parse_args()
    original=parent_sources(a.installed,a.vllm_root,a.archive,a.causal_archive)
    assert {n:gen.sha(b) for n,b in original.items()}==gen.PINS
    backend=gen.generated_backend(original['backend'].decode())
    env=gen.generated_env(original['env'].decode())
    assert gen.reversed_backend(backend)==original['backend'].decode()
    allocation=compile(ast.fix_missing_locations(ast.Module(body=gen.allocation_nodes(backend),type_ignores=[])),
                       '<actual-constructor-allocation>','exec')
    pool=compile(ast.fix_missing_locations(ast.Module(body=gen.pool_nodes(backend),type_ignores=[])),
                 '<actual-scratch-pool>','exec')
    count=0
    for selected,mq,heads,kvheads,dim,block,quant,cap in itertools.product(
            (16,32),(False,True),(24,32),(4,8),(128,256),(16,880,1760),(0,1),(80,89)):
        ev=NS(VLLM_TRITON_FP8_MQ3D_SEGMENTS=selected,VLLM_TRITON_FP8_MQ3D=mq,VLLM_TRITON_FP8_MQ3D_QMAX=8)
        ns=dict(envs=ev,KVQuantMode=NS(FP8_PER_TENSOR=1),NUM_PAR_SOFTMAX_SEGMENTS=16,
                current_platform=NS(is_device_capability=lambda c,cap=cap:c==cap))
        exec(gen.HELPER,ns)
        expected=32 if selected==32 and mq and (heads,kvheads,dim,block,quant,cap)==(24,4,256,880,1,89) else 16
        assert ns['_select_fp8_mq3d_segments'](heads,kvheads,dim,block,quant)==expected
        # Changing the registry later cannot change the frozen segment selection.
        ev.VLLM_TRITON_FP8_MQ3D_SEGMENTS=16 if selected==32 else 32
        assert ns['_select_fp8_mq3d_segments'](heads,kvheads,dim,block,quant)==expected
        # The installed pool, built from the installed source, with a torch that
        # only records what it was asked for.
        torch=FakeTorch()
        pow2=lambda x:1<<(x-1).bit_length()
        cfg=NS(model_config=NS(get_num_kv_heads=lambda _p,k=kvheads:k,get_head_size=lambda d=dim:d),
               parallel_config=NS(use_ubatching=False),
               compilation_config=NS(cudagraph_mode='PIECEWISE',cudagraph_capture_sizes=[]),
               scheduler_config=NS(max_num_batched_tokens=8192,max_num_seqs=4))
        pns=dict(torch=torch,dataclass=dataclass,VllmConfig=object,next_power_of_2=pow2,
                 logger=NS(info=lambda *a:None,warning=lambda *a:None),
                 _MQ3D_SCRATCH_POOL={},NUM_PAR_SOFTMAX_SEGMENTS=16,MIN_LAUNCH_GRID_SIZE_2D=128,MAX_QUERY_LEN_3D=16,
                 CUDAGraphMode=NS(FULL_AND_PIECEWISE='FULL_AND_PIECEWISE',
                                  FULL_DECODE_ONLY='FULL_DECODE_ONLY',FULL='FULL'))
        exec(pool,pns)
        obj=NS(num_heads_q=heads,num_heads_kv=kvheads,headdim=dim,block_size=block,
               seq_threshold_3D=32,vllm_config=cfg)
        ns.update(self=obj,torch=torch,device='cuda',kv_cache_spec=NS(kv_quant_mode=quant),
                  next_power_of_2=pow2,vllm_config=cfg,
                  mq3d_scratch_plan=pns['mq3d_scratch_plan'],
                  mq3d_scratch_acquire=pns['mq3d_scratch_acquire'])
        exec(allocation,ns)
        assert obj.num_par_softmax_segments==expected
        rows=min(8192,min(4,32)*16)
        # Both allocated buffers -- the third is an empty_like of the second --
        # carry the SELECTED count. A pool sized by the constant is what Z32
        # would have overrun.
        assert [v.shape for v in torch.calls]==[(rows,heads,expected,pow2(dim)),(rows,heads,expected)]
        assert all(v.dtype=='float32' and v.device.type=='cuda' for v in torch.calls)
        plan=pns['mq3d_scratch_plan'](cfg,heads,expected)
        assert plan.segments==expected
        assert plan.row_bytes==4*(heads*expected*pow2(dim)+2*heads*expected)
        assert pns['mq3d_scratch_plan'](cfg,heads).segments==16,'the default is the constant'
        count+=1
    invalid=0
    for text in ('','0','8','64','32.0','garbage'):
        ns=dict(os=NS(environ={gen.FLAG:text}))
        try:
            getter=eval('{'+gen.ENV_ENTRY.strip().rstrip(',')+'}',ns)[gen.FLAG]
            value=getter()
            exec(gen.HELPER,dict(envs=NS(VLLM_TRITON_FP8_MQ3D_SEGMENTS=value)))
        except (ValueError,NameError):invalid+=1
        else:raise AssertionError('Invalid explicit segment setting accepted')
    prior=dict(files=gen.FILES,installed=gen.PINS,delivery_revision='r2-prefill-only',
               flag='VLLM_TRITON_FP8_CAUSAL_FULL',default='0',frozen=True)
    data={'root/'+gen.FILES[n]:b for n,b in original.items()}
    data['causal/manifest.json']=json.dumps(prior).encode()
    root,archive=MemoryPath(data,'root'),MemoryPath(data,'causal')
    _,updated,_,continued,manifest=gen.prepare(root,archive)
    assert updated['unified']==original['unified'] and updated['helpers']==original['helpers']
    assert continued['installed']==manifest['installed']
    negative=0
    for name in gen.FILES:
        broken=dict(data);broken['root/'+gen.FILES[name]]+=b'\n'
        try:gen.prepare(MemoryPath(broken,'root'),MemoryPath(broken,'causal'))
        except AssertionError:negative+=1
        else:raise AssertionError('Source drift accepted')
    for field in ('files','installed'):
        broken=dict(data);r=copy.deepcopy(prior);r[field].pop('backend')
        broken['causal/manifest.json']=json.dumps(r).encode()
        try:gen.prepare(MemoryPath(broken,'root'),MemoryPath(broken,'causal'))
        except AssertionError:negative+=1
        else:raise AssertionError('Incomplete prior seal accepted')
    print(json.dumps(dict(status='FP8_DECODE_SEGMENTS_CPU_PASS',selector_and_allocation_cases=count,
        invalid_settings=invalid,seal_negative=negative,default_reverse_byte_identical=True,
        native_and_helpers_unchanged=True,installed=manifest['installed'],GPU=False)))


if __name__=='__main__':main()
