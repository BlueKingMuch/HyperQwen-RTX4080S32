"""Actual host AST and kernel byte identity; all native 3D routes reject r2."""
import argparse
import ast
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import fp8_causal_r2 as candidate
import test_fp8_causal_cpu as original


def main():
    p=argparse.ArgumentParser();p.add_argument('--r1-source',type=Path,required=True);a=p.parse_args()
    source=a.r1_source.read_text()
    result=candidate.generated_source(source)
    # Execute the retained 82 guard cases with only use_3d=False added to their
    # fixture. The compiled expression is the actual new HOST_GUARD.
    fn=next(n for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='guards')
    fn=copy.deepcopy(fn)
    fn.body.insert(1,ast.parse("base['use_3d']=False").body[0])
    ns=dict(candidate=candidate,NS=NS,itertools=itertools)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<r2-old-guards>','exec'),ns)
    old_count=ns['guards']()
    first=copy.deepcopy(fn.body[0]);env=dict(NS=NS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[first],type_ignores=[])),'<base>','exec'),env)
    base=env['base'];code=compile(candidate.HOST_GUARD.strip(),'<r2-actual-guard>','exec')
    rejected=0
    for enabled,n,q,dtype in itertools.product((False,True),range(1,5),(1,2,7,8,9,65,880,1024),('fp8','bf16','fp16','fp32')):
        values=dict(base,use_3d=True,_FP8_CAUSAL_FULL=enabled,num_seqs=n,max_seqlen_q=q,q=NS(dtype=dtype))
        exec(code,values);assert not values['_fp8_full_causal'];rejected+=1
    # Mixed prefix/suffix is selected independently using ACTUAL use_3d, not
    # original batch maximum query length or an inferred decode label.
    for mode in (False,True):
        for is3d in (False,True):
            values=dict(base,use_3d=is3d,_FP8_CAUSAL_FULL=mode)
            exec(code,values);assert values['_fp8_full_causal']==(mode and not is3d)
    print(json.dumps(dict(status='R2_CPU_PASS',source_sha256=candidate.r1.sha(result),
        unchanged_kernel_and_helper_count=len(candidate.r1.functions(source))-1,
        retained_2d_guards=old_count,rejected_3d_guards=rejected,mixed_route_cases=4)))


if __name__=='__main__':main()
