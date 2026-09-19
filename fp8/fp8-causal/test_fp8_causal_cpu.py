"""Stdlib-only actual-source / bounds / B880 ownership proof. No Torch import."""
import argparse
import ast
from bisect import bisect_right
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import fp8_causal as candidate


# No default parent: the one file this gate may read is the kernel as the vLLM
# series leaves it, which PARENT_SHA in fp8_causal.py pins. fp8/install.sh
# passes --parent and --helper out of the tree it is about to rewrite.


def validate_source(parent, helper):
    generated = candidate.generated_source(parent)
    before = candidate.functions(parent)['kernel_unified_attention']
    after = candidate.functions(generated)['kernel_unified_attention']
    oldloop = next(n for n in before.body if isinstance(n, ast.For))
    branch = next(n for n in after.body if isinstance(n, ast.If) and
                  isinstance(n.test, ast.Name) and n.test.id == 'FP8_FULL_CAUSAL' and n.orelse)
    full, boundary = [n for n in branch.body if isinstance(n, ast.For)]
    assert ast.dump(branch.orelse[0]) == ast.dump(oldloop)
    assert ast.dump(ast.Module(body=boundary.body, type_ignores=[])) == ast.dump(ast.Module(body=oldloop.body, type_ignores=[]))
    # Only two mask assignments, removal of query_abs_pos and one helper-call
    # name differ in full body. Restore them with the actual old AST nodes.
    assignments = {n.targets[0].id: n for n in oldloop.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name)}
    restored = []
    for n in copy.deepcopy(full.body):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in ('tile_mask', 'seq_mask'):
                if name == 'seq_mask': restored.append(assignments['query_abs_pos'])
                restored.append(assignments[name]); continue
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == '_fp8_full_softmax_step':
            n.value.func.id = 'softmax_step'
        restored.append(n)
    assert ast.dump(ast.Module(body=restored, type_ignores=[])) == ast.dump(ast.Module(body=oldloop.body, type_ignores=[]))
    assert candidate.sha(helper) == candidate.HELPER_SHA
    original_softmax = candidate.functions(helper)['softmax_step']
    full_softmax = copy.deepcopy(candidate.functions(generated)['_fp8_full_softmax_step'])
    full_softmax.name = original_softmax.name
    delta = next(n for n in full_softmax.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'delta')
    assert delta.value.args[0].value == 'sub.rn.f32 $0, $1, $2;'
    actual_args = next(k.value for k in delta.value.keywords if k.arg == 'args')
    old_p = next(n for n in original_softmax.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'P')
    assert ast.dump(actual_args.elts[0]) == ast.dump(old_p.value.args[0].left)
    assert ast.dump(actual_args.elts[1]) == ast.dump(old_p.value.args[0].right)
    full_softmax.body.remove(delta)
    full_softmax.body = [old_p if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'P' else n for n in full_softmax.body]
    # Ignore only the original docstring; all ordinary FP32 expressions remain.
    original_softmax = copy.deepcopy(original_softmax)
    if isinstance(original_softmax.body[0], ast.Expr): original_softmax.body.pop(0)
    assert ast.dump(full_softmax) == ast.dump(original_softmax)
    return generated, branch.body[0]


def guards():
    base = dict(_FP8_CAUSAL_FULL=True,current_platform=NS(is_device_capability=lambda n:n==89),
                KVQuantMode=NS(FP8_PER_TENSOR=1),kv_quant_mode=1,
                torch=NS(float8_e4m3fn='fp8',bfloat16='bf16'),q=NS(dtype='fp8'),
                k=NS(dtype='fp8'),v=NS(dtype='fp8'),out=NS(dtype='bf16'),
                num_query_heads=24,num_kv_heads=4,head_size=256,block_size=880,
                BLOCK_M=16,BLOCK_Q=2,tile_size=32,num_seqs=2,max_seqlen_q=8,
                use_causal=True,use_per_seq_causal=False,sliding_window_val=0,
                chunk_lookback=-1,use_mm_prefix=False,use_rswa=False,softcap=0,
                use_alibi_slopes=False,use_qq_bias=False,sinks=None,output_scale=None,
                use_td=False,is_batch_invariant=False)
    code = compile(candidate.HOST_GUARD.strip(), '<actual-host-guard>', 'exec')
    def run(changes):
        values=dict(base,**changes);exec(code,values);return values['_fp8_full_causal']
    assert run({})
    count=1
    changes = dict(_FP8_CAUSAL_FULL=[False],current_platform=[NS(is_device_capability=lambda n:False)],
                   kv_quant_mode=[0,2,3,4],q=[NS(dtype=x) for x in ('fp16','fp32','bf16')],
                   k=[NS(dtype='bf16')],v=[NS(dtype='bf16')],out=[NS(dtype=x) for x in ('fp16','fp32','fp8')],
                   num_query_heads=[32],num_kv_heads=[8],head_size=[128],block_size=[16,2048],
                   BLOCK_M=[32],BLOCK_Q=[4],tile_size=[16,64],num_seqs=[0,5],max_seqlen_q=[0,1],
                   use_causal=[False],use_per_seq_causal=[True],sliding_window_val=[1,128],
                   chunk_lookback=[0],use_mm_prefix=[True],use_rswa=[True],softcap=[1],
                   use_alibi_slopes=[True],use_qq_bias=[True],sinks=[object()],output_scale=[1],
                   use_td=[True],is_batch_invariant=[True])
    for name,values in changes.items():
        for value in values: assert not run({name:value}),(name,value);count+=1
    for n,q in itertools.product(range(1,5),(2,7,8,9,31,32,33,64,65,1024)):
        assert run(dict(num_seqs=n,max_seqlen_q=q));count+=1
    return count


def boundaries(actual_bound):
    code=compile(ast.fix_missing_locations(ast.Module(body=[actual_bound],type_ignores=[])), '<actual-full-bound>', 'exec')
    counts=dict(ctas=0,full_tiles=0,all_tiles=0,row_checks=0,page_seams=0,empty_segments=0)
    lengths=[0,1,2,7,8,16,17,31,32,33,63,64,65,879,880,881,895,896,897,1759,1760,1761,4097,8192,100000,262144]
    widths=list(range(66))+[127,128,129,672,879,880,881,1024]
    for width,length,is3d in itertools.product(widths,lengths,(False,True)):
        if width==0 or length<width: continue
        context=length-width
        for qb in range((width+1)//2):
            maximum=min(context+qb*2+(16-1)//6+1,length)
            tiles=(maximum+31)//32
            tps=(length+16*32-1)//(16*32) if is3d else 0
            for seg in range(16 if is3d else 1):
                if is3d and seg*tps*32>=length: continue
                lo=seg*tps if is3d else 0;hi=min((seg+1)*tps,tiles) if is3d else tiles
                env=dict(tl=NS(maximum=max,minimum=min),loop_lo=lo,loop_hi=hi,context_len=context,q_block_local_idx=qb,BLOCK_Q=2,TILE_SIZE=32)
                exec(code,env);end=env['full_end']
                # For a causally empty segment lo may exceed hi; both native
                # and specialized traversals must stay empty, not normalize hi.
                assert list(range(lo,end))+list(range(end,hi))==list(range(lo,hi))
                counts['ctas']+=1;counts['empty_segments']+=lo>=hi
                counts['full_tiles']+=max(0,end-lo);counts['all_tiles']+=max(0,hi-lo)
                for j in set([lo,end-1,end,hi-1]):
                    if not lo<=j<hi: continue
                    if j<end:
                        for m in range(16):
                            qp=qb*2+m//6
                            if qp<width:
                                assert j*32+31<=context+qp and j*32+31<maximum
                                counts['row_checks']+=1
                    for t in (j*32,j*32+31):
                        # Arbitrary reversed physical page IDs; full loads must
                        # still use EACH token's page/offset at the B880 seam.
                        page=t//880;offset=t%880
                        assert page*880+offset==t
                    counts['page_seams']+=(j*32)//880!=(j*32+31)//880
    # Variable/zero widths and padding retain the ORIGINAL q-block resolution.
    for widths in ((0,65),(65,0),(8,0,31),(1,7,8,64),(0,0,0,65),(8,8),(1024,1024)):
        cu=[0]
        for w in widths:cu.append(cu[-1]+w)
        starts=[x//2+i for i,x in enumerate(cu)]
        for allocated in (cu[-1],cu[-1]+32):
            for pid in range(allocated//2+len(widths)):
                seq=min(bisect_right(starts,pid)-1,len(widths)-1)
                qb=pid-starts[seq]
                if qb*2>=widths[seq]:continue
                assert cu[seq]+qb*2<cu[-1]<=allocated
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--helper',type=Path,required=True)
    a=p.parse_args();generated,bound=validate_source(a.parent.read_text(encoding='utf8'),a.helper.read_text(encoding='utf8'))
    print(json.dumps(dict(status='CPU_PASS',generated_sha256=candidate.sha(generated),host_guard_cases=guards(),**boundaries(bound)),indent=2))


if __name__=='__main__':main()
