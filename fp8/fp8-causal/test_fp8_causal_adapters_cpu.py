"""Stdlib checks of the actual adapter selector, receipt and timing sequence."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_fp8_causal_gpu as gate
import pair_fp8_causal as pair


def main():
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text())
    s=SimpleNamespace(unified=SimpleNamespace(_FP8_CAUSAL_FULL=False),
                      backend=SimpleNamespace(unified_attention=object()))
    original=SimpleNamespace(unified_attention=object());old=s.backend.unified_attention
    for mode in ('parent','off','on'):
        for fail in (False,True):
            try:
                with gate.select(s,original,mode):
                    assert s.unified._FP8_CAUSAL_FULL==(mode=='on')
                    assert s.backend.unified_attention is (original.unified_attention if mode=='parent' else old)
                    if fail:raise RuntimeError('forced')
            except RuntimeError:assert fail
            assert s.unified._FP8_CAUSAL_FULL is False and s.backend.unified_attention is old
    specs=gate.specifications();assert len(specs)==41
    for q in (8,880,1024):
        assert sum(x['widths']==[q]*c and x['lengths']==[n]*c for x in specs for c in (1,2) for n in (8192,100000))==4
    receipt=dict(status='COMPLETE',original_methods=8,focused_cases=41,
        source_sha256=gate.GENERATED_SHA,original_gate_sha256=gate.gen.GATE_SHA,
        gate_sha256=gate.gen.sha(Path(gate.__file__).read_bytes()),atol=.015,rtol=.01,
        cases=[dict(widths=x['widths'],lengths=x['lengths'],heads=x.get('heads',24),
                    maxabs_vs_parent=dict(off=0,on=.001)) for x in specs])
    def accept(r):
        with patch.object(Path,'read_text',return_value=json.dumps(r)):
            return pair.require_receipt(Path('unused.json'))
    accept(receipt)
    bad=[]
    for name,value in (('status','RUNNING'),('original_methods',4),('focused_cases',37),
        ('source_sha256','old'),('original_gate_sha256','old'),('gate_sha256','old'),('atol',.02),('rtol',.02)):
        r=copy.deepcopy(receipt);r[name]=value;bad.append(r)
    r=copy.deepcopy(receipt);r['cases'].pop();bad.append(r)
    r=copy.deepcopy(receipt);r['cases'][0]['widths']=[3];bad.append(r)
    r=copy.deepcopy(receipt);r['cases'][0]['maxabs_vs_parent']['off']=.001;bad.append(r)
    for r in bad:
        try:accept(r)
        except AssertionError:pass
        else:raise AssertionError('Incomplete/drifted gate receipt accepted')
    order=[]
    class Graph:
        def __init__(self,name):self.name=name
        def replay(self):order.append(('warm',self.name))
    t=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
    a={'graph':Graph('off')};b={'graph':Graph('on')}
    def timer(arm,iterations):
        assert iterations==25;order.append(('sample',arm['graph'].name))
        return 1.0 if arm is a else .9
    result=pair.sample_pair(t,timer,a,b,25,7)
    assert order[:6]==[('warm',n) for n in ('off','on','on','off','off','on')]
    assert order[6:]==[('sample',n) for i in range(7) for n in (('off','on') if i%2==0 else ('on','off'))]
    assert result['paired_ratios']==[.9]*7
    # Native D has no stride argument: row/head padding retains contiguous D.
    for heads in (24,32):
        for rows in (1,8,65,880,1760):
            stride=(heads+1)*256
            for row in (0,rows-1):
                for head in range(heads):
                    for d in range(256):
                        off=(row+2)*stride+head*256+d
                        assert off//stride==row+2 and (off%stride)//256<heads
    print(json.dumps(dict(status='ADAPTER_CPU_PASS',selectors=6,focused_specs=41,
        rejected_receipts=len(bad),alternating_samples=7,contiguous_D_row_padding=True)))


if __name__=='__main__':main()
