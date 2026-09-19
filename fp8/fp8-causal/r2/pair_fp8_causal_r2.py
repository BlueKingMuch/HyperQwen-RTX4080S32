"""Retained r1 pair procedure with only r2 source and per-real-route expectations."""
from pathlib import Path
import fp8_causal_r2 as candidate

PAIR_SHA='23cabc0a8f38cece189fb22bd63746b8a4dab6b7eec19c57460a7002198fe2bc'


def source():
    text=Path(__file__).resolve().parents[1].joinpath('pair_fp8_causal.py').read_text()
    assert candidate.r1.sha(text)==PAIR_SHA
    changes=(('import test_fp8_causal_gpu as gate','import test_fp8_causal_r2_gpu as gate'),
        ("assert rows and all(r['full']==(mode=='on') for r in rows),rows",
         "assert rows and all(r['full']==(mode=='on' and not r['is3d']) for r in rows),rows"))
    result=text
    for before,after in changes:
        assert result.count(before)==1;result=result.replace(before,after,1)
    reverse=result
    for before,after in reversed(changes):reverse=reverse.replace(after,before,1)
    assert reverse==text
    return result


if __name__=='__main__':
    exec(compile(source(),__file__,'exec'),dict(__name__='__main__',__file__=__file__))
