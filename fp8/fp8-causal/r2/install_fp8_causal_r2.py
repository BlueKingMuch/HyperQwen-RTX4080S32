"""Strict r1-image to r2 host-only patch; preserve both frozen ancestors."""
import argparse
import copy
import json
from pathlib import Path
import fp8_causal_r2 as candidate
import install_fp8_causal as base

ENV_INSTALLED_SHA='6761a702a2f5612f4e5615ba7a57587a46a30cf64a769bc36959fb09945fa978'


def prepare(root,archive):
    r1=candidate.r1
    prior=json.loads((archive/'manifest.json').read_text())
    expected=dict(base.PINS,unified=candidate.R1_SOURCE_SHA,env=ENV_INSTALLED_SHA)
    assert prior['parent']==base.PINS
    assert prior['files']==base.FILES and prior['installed']==expected
    assert prior['flag']==r1.FLAG and prior['default']=='0' and prior['frozen'] is True
    actual={name:r1.sha((root/path).read_bytes()) for name,path in base.FILES.items()}
    assert actual==expected,'r1 installed source drift'
    for name,digest in base.PINS.items():
        assert r1.sha((archive/'parent'/f'{name}.py').read_bytes())==digest,name
    source=(root/base.FILES['unified']).read_text()
    result=candidate.generated_source(source)
    updated=copy.deepcopy(prior)
    updated['installed']['unified']=r1.sha(result)
    updated.update(delivery_revision='r2-prefill-only',
        parent_component_unified=candidate.R1_SOURCE_SHA,
        generator_sha256=r1.sha(Path(candidate.__file__).read_bytes()),
        scope='SM89 native 2D only; plain causal FP8 H24/KV4/D256/B880; all 3D routes retain original loop')
    return source,result,prior,updated


def main():
    p=argparse.ArgumentParser();p.add_argument('--vllm-root',type=Path,default=Path('/usr/local/lib/python3.12/dist-packages/vllm'))
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    source,result,prior,updated=prepare(a.vllm_root,a.archive)
    if not a.check_only:
        frozen=a.archive/'r1';assert not frozen.exists(),'Never overwrite r1 archive'
        frozen.mkdir()
        (frozen/'unified.py').write_text(source)
        (frozen/'manifest.json').write_text(json.dumps(prior,indent=2)+'\n')
        (a.vllm_root/base.FILES['unified']).write_text(result)
        (a.archive/'manifest.json').write_text(json.dumps(updated,indent=2)+'\n')
    print(json.dumps(updated,indent=2))


if __name__=='__main__':main()
