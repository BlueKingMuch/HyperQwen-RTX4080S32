"""Host-only r2: measured native 3D decode remains on the original kernel path."""
import ast
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fp8_causal as r1

R1_SOURCE_SHA='b9e549320e30379b5755332d0a197a9cb28de2a06863b9e5ccb66f753f903a8e'
OLD='        _FP8_CAUSAL_FULL\n        and current_platform.is_device_capability(89)\n'
NEW='        _FP8_CAUSAL_FULL\n        and not use_3d\n        and current_platform.is_device_capability(89)\n'
HOST_GUARD=r1.HOST_GUARD.replace(OLD,NEW,1)
LAUNCH='    kernel_unified_attention[grid](\n'
LOG='''    if _fp8_full_causal:
        logger.info_once(
            "SYV_FP8_CAUSAL_FULL path=2D block=880 q_heads=24 maxq=%d",
            max_seqlen_q, scope="process",
        )

'''


def generated_source(source):
    assert r1.sha(source)==R1_SOURCE_SHA,'Exact r1 source required'
    assert source.count(OLD)==1
    result=source.replace(OLD,NEW,1)
    assert result.count(LAUNCH)==1
    result=result.replace(LAUNCH,LOG+LAUNCH,1)
    assert result.replace(LOG,'',1).replace(NEW,OLD,1)==source
    before,after=r1.functions(source),r1.functions(result)
    assert before.keys()==after.keys()
    for name in before.keys()-{'unified_attention'}:
        assert r1.source_block(source,before[name])==r1.source_block(result,after[name]),name
        assert ast.dump(before[name])==ast.dump(after[name]),name
    original_guard=next(n for n in ast.walk(after['unified_attention']) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_fp8_full_causal' for t in n.targets))
    assert isinstance(original_guard.value,ast.BoolOp) and isinstance(original_guard.value.op,ast.And)
    assert any(isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.Not) and isinstance(n.operand,ast.Name)
               and n.operand.id=='use_3d' for n in original_guard.value.values)
    compile(result,'<fp8-causal-r2-host-only>','exec')
    return result
