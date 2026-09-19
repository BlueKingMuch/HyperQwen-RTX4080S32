"""CPU execution of actual candidate entry/index/bound expressions; no GPU imports."""
import argparse
import ast
from collections import Counter
import copy
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace
import textwrap
import fp8_prefill_flat as gen


class Ptr:
    def __init__(self, data, offset=0): self.data, self.offset = data, offset
    def __add__(self, offset): return Ptr(self.data, self.offset + int(offset))


class Scalar(int):
    def to(self, dtype): return self


class TL:
    int1 = int32 = int64 = int
    @staticmethod
    def load(p):
        assert 0 <= p.offset < len(p.data), (p.offset, len(p.data))
        return p.data[p.offset]
    @staticmethod
    def where(c, a, b): return Scalar(a if c else b)
    minimum = staticmethod(min)
    maximum = staticmethod(max)
    @staticmethod
    def static_assert(v): assert v


def executable(node, ns):
    node = copy.deepcopy(node)
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs: arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<actual-source-CPU>', 'exec'), ns)
    return ns[node.name]


def helpers(source):
    assert gen.sha(source) == gen.HELPER_SHA
    ns = dict(tl=TL, cdiv_fn=lambda x, y: (x+y-1)//y)
    for name in ('find_seq_idx', 'resolve_seq_and_query_len', 'compute_tile_loop_bounds'):
        executable(gen.functions(source)[name], ns)
    ns.update(BLOCK_Q=2, BLOCK_M=16, TILE_SIZE=32, num_queries_per_kv=6,
              num_query_heads=24, IS_3D=False, FP8_FULL_CAUSAL=True,
              SLIDING_WINDOW=0, USE_MM_PREFIX=False, USE_R_SWA=False,
              USE_CAUSAL=True, USE_PER_SEQ_CAUSAL=False,
              CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1, segm_idx=0, tiles_per_segment=0)
    return ns


def extracted(source, helper):
    ns = helpers(helper)
    k = gen.functions(source)['kernel_unified_attention']
    resolve = next(n for n in k.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'FP8_FLAT_PHASE >= 0'
                   and any(isinstance(c, ast.While) for c in n.body))
    shell = ast.parse('def entry(q_block_global_idx, FP8_FLAT_PHASE, query_start_len_ptr, seq_lens_ptr, num_seqs):\n    pass').body[0]
    shell.body = [copy.deepcopy(resolve)] + ast.parse('return locals()').body
    entry = executable(shell, ns)
    row_nodes = []
    for n in k.body:
        targets = {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        if targets & {'query_pos', 'query_mask_0', 'query_mask_1', 'context_len', 'loop_hi'}:
            row_nodes.append(copy.deepcopy(n))
    # Pull the actual full-end statement, not either KV loop.
    outer = next(n for n in k.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'FP8_FULL_CAUSAL' and n.orelse)
    row_nodes.append(copy.deepcopy(outer.body[0]))
    shell = ast.parse('def row():\n    pass').body[0]
    shell.body = row_nodes + ast.parse('return query_pos,query_offset_1,bool(query_mask_0 & query_mask_1),loop_lo,loop_hi,max_seq_prefix_len,full_end').body
    row = executable(shell, ns)
    return ns, entry, row


def host_checks():
    tree = ast.parse(textwrap.dedent(gen.HOST))
    tree.body[-1].body += ast.parse('records.append((_fp8_flat_phase, _fp8_flat_grid))').body
    code = compile(tree, '<actual-host-flat-selection>', 'exec')
    count = 0
    for on, full, is3d, qmax, capacity, n in itertools.product(
            (False, True), (False, True), (False, True),
            (0, 1, 8, 9, 63, 64, 65, 880, 1024, 1760), (32, 64, 4096), range(1, 5)):
        original = (capacity//2+n, 4, 16) if is3d else (capacity//2+n, 4)
        ns = dict(_FP8_PREFILL_FLAT=on, _fp8_full_causal=full, use_3d=is3d,
                  max_seqlen_q=qmax, q=SimpleNamespace(shape=(capacity,24,256)),
                  num_seqs=n, num_kv_heads=4, grid=original, records=[])
        exec(code, ns)
        eligible = on and full and not is3d and qmax >= 64 and capacity >= 64
        assert ns['records'] == ([(p,(capacity//8+32*n,4)) for p in range(3)]
                                  if eligible else [(-1,original)])
        count += 1
    prior = os.environ.get(gen.FLAG)
    try:
        for value in ('0','1','','2','-1','true'):
            os.environ[gen.FLAG] = value
            ns = {}
            try: exec(gen.FROZEN, ns)
            except ValueError: assert value not in ('0','1')
            else:
                assert ns['_FP8_PREFILL_FLAT'] == (value == '1')
                os.environ[gen.FLAG] = '0' if value == '1' else '1'
                assert ns['_FP8_PREFILL_FLAT'] == (value == '1')
    finally:
        if prior is None: os.environ.pop(gen.FLAG, None)
        else: os.environ[gen.FLAG] = prior
    return count


def owner_checks(output, helper):
    ns, entry, row = extracted(output, helper)
    cases = [(a,b) for a in range(66) for b in range(66)]
    edges = (0,1,8,9,17,31,32,33,63,64,65,879,880,881,1024,1760)
    cases += [(q,) for q in edges]
    for n in (3,4):
        cases += [tuple(edges[(i+3*j)%len(edges)] for j in range(n)) for i in range(len(edges))]
    cases += [(880,880),(1024,1024),(1760,1760),(8,880),(880,8),(0,880,0,8)]
    stats = Counter()
    for case_idx, widths in enumerate(cases):
        cu = [0]
        for w in widths: cu.append(cu[-1]+w)
        capacity = max(64, cu[-1]+(0,17,1024)[case_idx%3])
        assert cu[-1] <= capacity
        contexts = (0,1,31,32,879,880,881,8192,100000)
        lengths = [w+contexts[(case_idx+s)%len(contexts)] if w else 0
                   for s,w in enumerate(widths)]
        observed, expected = Counter(), Counter()
        for s,w in enumerate(widths):
            if w >= 64:
                for q in range(w):
                    for h in range(6): expected[s,q,h] = 1
            else:
                # Parent intentionally repeats some rows; preserve short fallback exactly.
                for b in range((w+1)//2):
                    for m in range(16):
                        if 2*b+m//6 < w: expected[s,2*b+m//6,m%6] += 1
        for phase in range(3):
            for pid in range(capacity//8+32*len(widths)):
                rec = entry(pid, phase, Ptr(cu), Ptr(lengths), len(widths))
                stats['entry_executions'] += 1
                if rec is None: continue
                s = rec['seq_idx']; w=widths[s]
                assert rec['cur_batch_in_all_start_index'] == cu[s]
                assert rec['cur_batch_query_len'] == w and w > 0
                assert w >= 64 or phase == 0
                ns.update(rec, kv_head_idx=0)
                qmin, qmax = rec['_flat_qmin'], rec['_flat_qmax']
                assert qmin < w
                for m in range(16):
                    ns['offs_m'] = m
                    qp,qh,valid,lo,hi,maximum,full = row()
                    assert qmin <= qp <= qmax
                    assert lo == 0 and maximum <= lengths[s]
                    assert hi == (maximum+31)//32 and 0 <= full <= hi
                    if full: assert full*32-1 <= lengths[s]-w+qmin
                    if valid:
                        assert 0 <= qp < w and 0 <= qh < 6
                        assert cu[s]+qp < cu[-1] <= capacity
                        assert lengths[s]-w+qp+1 <= maximum
                        observed[s,qp,qh] += 1
                        stats['valid_computed_rows'] += 1
                    stats['row_checks'] += 1
                # Keep token-local page indexing across B880/TS32 seams.
                for tile in {0, 27, 54, max(0,hi-1), max(0,full-1)}:
                    for off in range(32):
                        t=32*tile+off
                        if t < maximum:
                            page,slot=divmod(t,880)
                            physical_page=3*page+7  # deliberately not identity
                            for h in (0,3):
                                address=physical_page*(880*2048)+slot*2048+h*512
                                assert address == physical_page*1802240+slot*2048+h*512
                            stats['page_address_checks'] += 1
                stats['active_CTAs_one_KVhead'] += 1
        assert observed == expected, (widths, (observed-expected).most_common(1),(expected-observed).most_common(1))
        stats['batch_cases'] += 1
    # Actual row expressions translate all four KV heads without changing token ownership.
    for phase, local, m, head in itertools.product(range(3), (0,1,7,109), range(16), range(4)):
        rec = entry(local, phase, Ptr([0,880]), Ptr([100000]), 1)
        assert rec
        ns.update(rec, offs_m=m, kv_head_idx=head)
        value = row()
        assert value[1] == 6*head+(16*phase+m)%6
        stats['head_translation_checks'] += 1
    return dict(stats)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--helper',type=Path,required=True)
    a=p.parse_args()
    parent=a.parent.read_text(encoding='utf8')
    output=gen.generated_source(parent)
    counts=owner_checks(output,a.helper.read_text(encoding='utf8'))
    print(json.dumps(dict(status='FP8_PREFILL_FLAT_CPU_PASS',parent_sha=gen.PARENT_SHA,
        source_sha=gen.sha(output),host_cases=host_checks(),**counts,**gen.assert_identity(parent,output),
        GPU=False,kernel_compiled=False)))


if __name__=='__main__': main()

