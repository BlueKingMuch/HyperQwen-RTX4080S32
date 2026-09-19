"""Native full-forward gate. Root GPU owner only; no serving lifecycle actions."""
import argparse
import os
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import fp8_causal as gen

# Imported by the CPU gate for its case table and its oracles. Run as a script it
# additionally wants a GPU and the archive fp8/install.sh wrote, and nothing in
# this repository runs it that way -- so the archive is read from the same
# variable install.sh uses rather than from a path that would not exist.
ARCHIVE = Path(os.environ.get('FP8_ARCHIVE', '/opt/fp8')) / 'causal'

GENERATED_SHA='fcbb5d45948d61b0f6673a111aa924b0f2f37b34222228b6ec9b977710e72159'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def bootstrap(path):
    assert gen.sha(path.read_bytes())==gen.GATE_SHA,'Original gate drift'
    suite=load('original_native_fp8_gate',path)
    manifest=json.loads((ARCHIVE/'manifest.json').read_text())
    root=Path(suite.unified.__file__).parents[3]
    for name,relative in manifest['files'].items():
        assert gen.sha((root/relative).read_bytes())==manifest['installed'][name],name
    assert gen.sha(Path(suite.unified.__file__).read_bytes())==GENERATED_SHA
    parent_path=ARCHIVE/'parent/unified.py'
    assert gen.sha(parent_path.read_bytes())==gen.PARENT_SHA
    original=load('archived_original_native_fp8',parent_path)
    assert suite.torch.cuda.get_device_capability()==(8,9)
    suite.UNIFIED_SHA=GENERATED_SHA  # Only expected source pin; all test methods unchanged.
    return suite,original


class Recorder:
    def __init__(self,kernel,rows):self.kernel,self.rows=kernel,rows
    def __getitem__(self,grid):
        launch=self.kernel[grid]
        def run(*args,**kw):
            self.rows.append(dict(full=bool(kw['FP8_FULL_CAUSAL']),is3d=bool(kw['IS_3D']),
                tile=kw['TILE_SIZE'],segments=kw['NUM_SEGMENTS_PER_SEQ'],heads=kw['num_query_heads'],
                k_strides=[kw[f'stride_k_cache_{i}'] for i in range(4)]))
            return launch(*args,**kw)
        return run


def create_case(s,widths,lengths,heads=24,block=880,dtype='float8_e4m3fn',padded=True,causal=True):
    t=s.torch;t.manual_seed(19063+sum(widths)+sum(lengths))
    with patch.object(s,'HEADS',heads),patch.object(s,'BLOCK',block):
        batch=s.Batch(widths,lengths,[w>8 for w in widths]);builder=s.make_builder();impl=s.make_impl()
    # Preserve logical bytes but use actual backend NHD physical ownership.
    batch.cache=batch.cache.transpose(1,2).contiguous().transpose(1,2)
    key=batch.cache.transpose(1,2)[...,:256]
    assert tuple(key.stride())==(block*2048,2048,512,1)
    # Unowned physical page and tail slots carry poison: a removed memory mask
    # must be justified by full_end, never by a hidden zero-padded cache.
    nan=t.tensor(float('nan'),dtype=t.bfloat16,device='cuda').to(t.float8_e4m3fn)
    batch.cache[0].copy_(nan.expand_as(batch.cache[0]))
    table=batch.table.cpu()
    for row,length in enumerate(lengths):
        count=(length+block-1)//block;tail=length%block
        if count and tail:
            view=batch.cache[int(table[row,count-1]),:,tail:,:]
            view.copy_(nan.expand_as(view))
    values=batch.q.to(getattr(t,dtype))
    capacity=batch.total+(16 if padded else 0)
    qstore=t.empty((capacity,heads+1,256),dtype=values.dtype,device='cuda')
    # uint8 initialization also works for FP8 and retains guard bit patterns.
    qstore.view(t.uint8).fill_(0x35)
    q=qstore[:batch.total,:heads,:];q.copy_(values);batch.q=q
    assert q.stride(-1)==1,'Native ABI has implicit contiguous D'
    batch.common.causal=causal
    assert batch.common.num_actual_tokens==sum(widths)<=batch.q.shape[0]
    assert int(batch.cu[-1].item())<=batch.q.shape[0]
    metadata=builder.build(0,batch.common)
    return dict(batch=batch,builder=builder,impl=impl,metadata=metadata,heads=heads,
                qstore=qstore,qbits=qstore.view(t.uint8).clone(),widths=widths,lengths=lengths,
                expected_on=(heads==24 and block==880 and dtype=='float8_e4m3fn' and causal and max(widths)>1))


def output_storage(s,case):
    t=s.torch;n=case['batch'].total;h=case['heads']
    storage=t.full((n+4,h+1,256),-123.,dtype=t.bfloat16,device='cuda')
    return storage,storage[2:2+n,:h,:]


def guards(s,case,storage):
    t=s.torch
    assert bool(t.all(storage[:2]==-123)) and bool(t.all(storage[-2:]==-123))
    assert bool(t.all(storage[2:-2,-1,:]==-123)),'Output row-padding head overwritten'
    assert t.equal(case['qstore'].view(t.uint8),case['qbits']),'Q or its padding changed'


@contextmanager
def select(s,original,arm,rows=None):
    if arm=='parent':
        with patch.object(s.backend,'unified_attention',original.unified_attention):yield
    else:
        with patch.object(s.unified,'_FP8_CAUSAL_FULL',arm=='on'):
            if rows is None:yield
            else:
                with patch.object(s.unified,'kernel_unified_attention',Recorder(s.unified.kernel_unified_attention,rows)):yield


def invoke(s,case,out):
    b=case['batch']
    return case['impl'].forward(b.layer,b.q,None,None,b.cache,case['metadata'],out)


def check_case(s,original,case):
    t=s.torch;results={};routes={};maxdiff={}
    for arm in ('parent','off','on'):
        s.poison(case['builder']);storage,out=output_storage(s,case);rows=[]
        with select(s,original,arm,rows):assert invoke(s,case,out) is out
        t.cuda.synchronize();guards(s,case,storage)
        assert bool(t.isfinite(out).all()),(arm,case['widths'],case['lengths'])
        results[arm]=out.clone();routes[arm]=rows
    t.testing.assert_close(results['off'],results['parent'],atol=0,rtol=0)
    t.testing.assert_close(results['on'],results['parent'],atol=s.ATOL if case['expected_on'] else 0,
                           rtol=s.RTOL if case['expected_on'] else 0)
    assert routes['off'] and not any(x['full'] for x in routes['off'])
    assert bool(any(x['full'] for x in routes['on']))==case['expected_on'],routes
    for arm in ('off','on'):maxdiff[arm]=float((results[arm].float()-results['parent'].float()).abs().max())
    return dict(widths=case['widths'],lengths=case['lengths'],heads=case['heads'],
                active=case['expected_on'],maxabs_vs_parent=maxdiff,routes=routes)


def specifications():
    rows=[dict(widths=[q],lengths=[q+32]) for q in (2,8,9,17,31,32,33,63,64,65)]
    rows += [dict(widths=[17],lengths=[n]) for n in (879,880,881,895,896,897,1761)]
    rows += [dict(widths=w,lengths=l) for w,l in (([1,7,8,64],[881,895,896,1761]),
        ([8,65],[8192,897]),([65,8],[897,8192]),([0,65],[0,897]),([65,0],[897,0]))]
    rows += [dict(widths=[q]*c,lengths=[n]*c) for q in (8,880,1024) for c in (1,2) for n in (8192,100000)]
    rows += [dict(widths=[8],lengths=[881],dtype=d) for d in ('bfloat16','float16','float32')]
    rows += [dict(widths=[8],lengths=[881],heads=32),dict(widths=[8],lengths=[881],block=16),
             dict(widths=[8],lengths=[881],causal=False),dict(widths=[1],lengths=[881])]
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-gate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();s,original=bootstrap(a.original_gate)
    report=dict(status='RUNNING',source_sha256=GENERATED_SHA,gate_sha256=gen.sha(Path(__file__).read_bytes()),
                original_gate_sha256=gen.GATE_SHA,atol=s.ATOL,rtol=s.RTOL,original_methods=0,cases=[])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for enabled in (False,True):
        tests=unittest.defaultTestLoader.loadTestsFromTestCase(s.MixedFp8GPU)
        with patch.object(s.unified,'_FP8_CAUSAL_FULL',enabled):
            result=unittest.TextTestRunner(verbosity=2).run(tests)
        assert result.wasSuccessful() and result.testsRun==4,'Unchanged original FP8 gate failed'
        report['original_methods']+=result.testsRun;save()
    with s.torch.inference_mode():
        for spec in specifications():
            case=create_case(s,**spec);row=check_case(s,original,case)
            report['cases'].append(row);print(json.dumps(row),flush=True);save()
            del case;s.torch.cuda.empty_cache()
    report['status']='COMPLETE';report['focused_cases']=len(report['cases']);save()
    print('FP8_CAUSAL_GPU_COMPLETE',json.dumps(dict(original_methods=8,focused_cases=len(report['cases']))))


if __name__=='__main__':main()
