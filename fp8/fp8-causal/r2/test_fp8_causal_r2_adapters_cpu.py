"""Per-real-route contract and unchanged timing bodies; stdlib only."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch
import fp8_causal_r2 as candidate
import test_fp8_causal_r2_gpu as gate
import pair_fp8_causal_r2 as pair


def main():
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text())
    count=0
    for eligible in (False,True):
        for enabled in (False,True):
            for rows in ([False],[True],[True,False],[False,True],[True,True]):
                actual=gate.expected_flags(dict(expected_on=eligible),[dict(is3d=x) for x in rows],enabled)
                assert actual==[enabled and eligible and not x for x in rows]
                assert all(not a for a,x in zip(actual,rows) if x);count+=1
    source=pair.source();ns=dict(__name__='cpu_proof',__file__=pair.__file__)
    exec(compile(source,'<actual-r2-pair>','exec'),ns)
    before=ast.parse(Path(pair.__file__).resolve().parents[1].joinpath('pair_fp8_causal.py').read_text())
    after=ast.parse(source)
    names={'capture','sample_pair','time_graph','verify','require_receipt'}
    def functions(tree):return {n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    assert functions(before)==functions(after)
    specs=gate.specifications()
    receipt=dict(status='COMPLETE',original_methods=8,focused_cases=41,source_sha256=gate.GENERATED_SHA,
        original_gate_sha256=gate.gen.GATE_SHA,gate_sha256=gate.gen.sha(Path(gate.__file__).read_bytes()),
        atol=.015,rtol=.01,cases=[dict(widths=x['widths'],lengths=x['lengths'],heads=x.get('heads',24),
            maxabs_vs_parent=dict(off=0,on=0)) for x in specs])
    def accept(r):
        with patch.object(Path,'read_text',return_value=json.dumps(r)):return ns['require_receipt'](Path('receipt.json'))
    accept(receipt);bad=[]
    for key,value in (('status','RUNNING'),('source_sha256',candidate.R1_SOURCE_SHA),('gate_sha256',gate.R1_GATE_SHA),
                      ('focused_cases',37),('original_methods',4),('atol',.02),('rtol',.02)):
        r=copy.deepcopy(receipt);r[key]=value;bad.append(r)
    for r in bad:
        try:accept(r)
        except AssertionError:pass
        else:raise AssertionError('Bad receipt accepted')
    print(json.dumps(dict(status='R2_ADAPTER_CPU_PASS',actual_route_cases=count,
        unchanged_timing_and_verify_functions=5,rejected_receipts=len(bad))))


if __name__=='__main__':main()
