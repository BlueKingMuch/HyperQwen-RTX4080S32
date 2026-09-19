"""R2 actual per-launch gate: mixed 3D prefix OFF, eligible 2D suffix ON."""
import argparse
import os
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import fp8_causal_r2 as candidate
import test_fp8_causal_gpu as r1

# Imported by the CPU gate for its case table and its oracles. Run as a script it
# additionally wants a GPU and the archive fp8/install.sh wrote, and nothing in
# this repository runs it that way -- so the archive is read from the same
# variable install.sh uses rather than from a path that would not exist.
ARCHIVE = Path(os.environ.get('FP8_ARCHIVE', '/opt/fp8')) / 'causal'

gen=candidate.r1
GENERATED_SHA='e940e8f52c0b595a37bb297acdbac79b9e221a7ffe79a4e5117f01a27e628f2f'
R1_GATE_SHA='81298dbef77972e70647ec7554fb99e59fbb07c68202cc6ec5607c845e738eb0'
specifications=r1.specifications
create_case=r1.create_case
output_storage=r1.output_storage
guards=r1.guards
select=r1.select
invoke=r1.invoke


def bootstrap(path):
    assert gen.sha(Path(r1.__file__).read_bytes())==R1_GATE_SHA
    r1.GENERATED_SHA=GENERATED_SHA
    s,original=r1.bootstrap(path)
    archive=ARCHIVE
    manifest=json.loads((archive/'manifest.json').read_text())
    assert manifest['delivery_revision']=='r2-prefill-only'
    assert manifest['parent_component_unified']==candidate.R1_SOURCE_SHA
    assert gen.sha(candidate.generated_source((archive/'r1/unified.py').read_text()))==GENERATED_SHA
    return s,original


def expected_flags(case,rows,enabled):
    return [bool(enabled and case['expected_on'] and not row['is3d']) for row in rows]


def check_case(s,original,case):
    t=s.torch;results={};routes={}
    for arm in ('parent','off','on'):
        s.poison(case['builder']);storage,out=output_storage(s,case);rows=[]
        with select(s,original,arm,rows):assert invoke(s,case,out) is out
        t.cuda.synchronize();guards(s,case,storage)
        assert bool(t.isfinite(out).all()),(arm,case['widths'],case['lengths'])
        results[arm]=out.clone();routes[arm]=rows
    t.testing.assert_close(results['off'],results['parent'],atol=0,rtol=0)
    assert routes['off'] and len(routes['off'])==len(routes['on'])
    assert [{k:v for k,v in r.items() if k!='full'} for r in routes['off']]==[
           {k:v for k,v in r.items() if k!='full'} for r in routes['on']]
    expected=expected_flags(case,routes['off'],True)
    assert [r['full'] for r in routes['off']]==[False]*len(expected)
    assert [r['full'] for r in routes['on']]==expected,(case['widths'],routes)
    assert all(not r['full'] for r in routes['on'] if r['is3d'])
    active=any(expected)
    t.testing.assert_close(results['on'],results['parent'],atol=s.ATOL if active else 0,rtol=s.RTOL if active else 0)
    return dict(widths=case['widths'],lengths=case['lengths'],heads=case['heads'],active=active,
        expected_flags=expected,maxabs_vs_parent={arm:float((results[arm].float()-results['parent'].float()).abs().max())
            for arm in ('off','on')},routes=routes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-gate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();s,original=bootstrap(a.original_gate)
    assert not a.out.exists(),'Preserve existing gate receipts'
    report=dict(status='RUNNING',revision='r2-prefill-only',source_sha256=GENERATED_SHA,
        gate_sha256=gen.sha(Path(__file__).read_bytes()),original_gate_sha256=gen.GATE_SHA,
        retained_r1_adapter_sha256=R1_GATE_SHA,atol=s.ATOL,rtol=s.RTOL,original_methods=0,cases=[])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for enabled in (False,True):
        tests=unittest.defaultTestLoader.loadTestsFromTestCase(s.MixedFp8GPU)
        with patch.object(s.unified,'_FP8_CAUSAL_FULL',enabled):
            result=unittest.TextTestRunner(verbosity=2).run(tests)
        assert result.wasSuccessful() and result.testsRun==4
        report['original_methods']+=result.testsRun;save()
    with s.torch.inference_mode():
        for spec in specifications():
            case=create_case(s,**spec);row=check_case(s,original,case)
            report['cases'].append(row);print(json.dumps(row),flush=True);save()
            del case;s.torch.cuda.empty_cache()
    report['status']='COMPLETE';report['focused_cases']=len(report['cases']);save()
    print('FP8_CAUSAL_R2_GPU_COMPLETE')


if __name__=='__main__':main()
