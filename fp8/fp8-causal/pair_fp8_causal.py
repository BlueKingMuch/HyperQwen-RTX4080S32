"""Four same-source native FP8 full-forward pairs; no model-throughput claim.

capture/sample_pair/time_graph retain the reviewed geometry/decode driver ASTs.
Q is already quantized in both arms, as in the actual attention input ABI.
"""
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import statistics
import test_fp8_causal_gpu as gate


def capture(torch, invoke, warmups, iterations):
    for _ in range(warmups):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            invoke()
    graph.replay()
    torch.cuda.synchronize()
    return dict(graph=graph)


def sample_pair(torch, timer, reference, candidate, iterations, samples):
    graphs = {'reference': reference, 'candidate': candidate}
    for round_ in range(3):
        for name in (('reference', 'candidate') if round_%2 == 0 else ('candidate', 'reference')):
            graphs[name]['graph'].replay()
            torch.cuda.synchronize()
    values = {name: [] for name in graphs}
    for sample in range(samples):
        for name in (('reference', 'candidate') if sample%2 == 0 else ('candidate', 'reference')):
            values[name].append(timer(graphs[name], iterations))
    ratios = [on/off for on, off in zip(values['candidate'], values['reference'])]
    medians = {name: statistics.median(data) for name, data in values.items()}
    return dict(milliseconds_per_call=values, median_ms=medians,
                sample_sd_ms={name: statistics.stdev(data) for name, data in values.items()},
                candidate_over_reference_median=medians['candidate']/medians['reference'],
                paired_ratios=ratios, paired_ratio_mean=statistics.mean(ratios),
                paired_ratio_sample_sd=statistics.stdev(ratios))


def time_graph(arm, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    arm['graph'].replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def require_receipt(path):
    r=json.loads(path.read_text())
    assert r['status']=='COMPLETE' and r['original_methods']==8 and r['focused_cases']==41
    assert r['source_sha256']==gate.GENERATED_SHA and r['original_gate_sha256']==gate.gen.GATE_SHA
    assert r['gate_sha256']==gate.gen.sha(Path(gate.__file__).read_bytes())
    assert r['atol']==.015 and r['rtol']==.01 and len(r['cases'])==41
    expected=gate.specifications()
    assert [(x['widths'],x['lengths'],x['heads']) for x in r['cases']]==[(x['widths'],x['lengths'],x.get('heads',24)) for x in expected]
    assert all(x['maxabs_vs_parent']['off']==0 for x in r['cases'])
    return r


def verify(s,case,arms,reference):
    t=s.torch
    for mode,arm in arms.items():
        arm['graph'].replay();t.cuda.synchronize();gate.guards(s,case,arm['storage'])
        t.testing.assert_close(arm['output'],reference,atol=s.ATOL if mode=='on' else 0,
                               rtol=s.RTOL if mode=='on' else 0)
    return dict(maxabs=float((arms['on']['output'].float()-reference.float()).abs().max()),
                Q_unchanged=True,strided_output_guards=True,parent_reference=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-gate',type=Path,required=True)
    p.add_argument('--gates-log',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--query',type=int,choices=(8,880,1024),default=880);a=p.parse_args()
    require_receipt(a.gates_log);s,original=gate.bootstrap(a.original_gate);t=s.torch
    # The imported timer body is byte-for-byte/AST equivalent to the established
    # global-torch timer; only its module namespace is bound here.
    globals()['torch']=t
    assert not a.out.exists(),'Never overwrite retained timings'
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='RUNNING',source_sha256=gate.GENERATED_SHA,gate_sha256=gate.gen.sha(a.gates_log.read_bytes()),
        pair_script_sha256=gate.gen.sha(Path(__file__).read_bytes()),query_each=a.query,
        warmups=5,iterations_per_graph=25,alternating_graph_warm_rounds=3,paired_samples=7,
        scope='actual TritonAttentionImpl.forward; prequantized FP8 Q and ready metadata/cache; no KV writes',
        model_claim=False,cache_policy='same resident cache per paired arm; no flush',cases=[])
    with t.inference_mode():
        for length in (8192,100000):
            for batch in (1,2):
                case=gate.create_case(s,[a.query]*batch,[length]*batch,padded=False)
                reference_storage,reference=gate.output_storage(s,case)
                with gate.select(s,original,'parent'):gate.invoke(s,case,reference)
                t.cuda.synchronize();gate.guards(s,case,reference_storage)
                arms={};routes={}
                for mode in ('off','on'):
                    storage,out=gate.output_storage(s,case);rows=[]
                    with gate.select(s,original,mode,rows):gate.invoke(s,case,out)
                    assert rows and all(r['full']==(mode=='on') for r in rows),rows
                    def invoke(mode=mode,out=out):
                        with gate.select(s,original,mode):gate.invoke(s,case,out)
                    arm=capture(t,invoke,5,25);arm.update(storage=storage,output=out)
                    arms[mode]=arm;routes[mode]=rows
                before=verify(s,case,arms,reference);windows=[]
                def timer(arm,iterations):
                    start=datetime.now(timezone.utc).isoformat();ms=time_graph(arm,iterations)
                    windows.append(dict(arm='off' if arm is arms['off'] else 'on',start=start,
                        end=datetime.now(timezone.utc).isoformat(),ms=ms,replay_calls=iterations))
                    return ms
                timing=sample_pair(t,timer,arms['off'],arms['on'],25,7)
                after=verify(s,case,arms,reference)
                row=dict(batch=batch,context_each=length,query_each=a.query,before=before,after=after,
                         routes=routes,whole_wrapper=timing,timing_windows=windows)
                report['cases'].append(row);a.out.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(batch=batch,context=length,timing=timing)),flush=True)
                del case,arms,arm,out,storage,reference,reference_storage,invoke
                gc.collect();t.cuda.empty_cache()
    report['status']='COMPLETE';a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('FP8_CAUSAL_PAIR_COMPLETE')


if __name__=='__main__':main()
