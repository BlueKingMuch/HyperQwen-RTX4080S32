"""Install only from the exact Stable native sources; preserve original bytes."""
import argparse
import json
from pathlib import Path
import fp8_causal as candidate

# PARENT: the tree after patches/series and kvarn/install.sh, which is what PINS
# below is of. A docker image id stood here; on this tree the parent is not an
# image, it is four files, and their sha256 is the whole of the claim.
FILES=dict(unified='v1/attention/ops/triton_unified_attention.py',
           helpers='v1/attention/ops/triton_attention_helpers.py',
           backend='v1/attention/backends/triton_attn.py',env='envs.py')
PINS=dict(unified=candidate.PARENT_SHA,helpers=candidate.HELPER_SHA,
          backend=candidate.BACKEND_SHA,env=candidate.ENV_SHA)


def prepare(root):
    original={name:(root/path).read_bytes() for name,path in FILES.items()}
    assert {name:candidate.sha(data) for name,data in original.items()}==PINS,'Stable source drift'
    updated=dict(original)
    updated['unified']=candidate.generated_source(original['unified'].decode()).encode()
    updated['env']=candidate.generated_env(original['env'].decode()).encode()
    return original,updated


def main():
    p=argparse.ArgumentParser();p.add_argument('--vllm-root',type=Path,default=Path('/usr/local/lib/python3.12/dist-packages/vllm'))
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    original,updated=prepare(a.vllm_root)
    manifest=dict(flag=candidate.FLAG,default='0',frozen=True,
                  parent=PINS,installed={n:candidate.sha(b) for n,b in updated.items()},
                  files=FILES,generator_sha256=candidate.sha(Path(candidate.__file__).read_bytes()),
                  scope='SM89 plain causal FP8 per-tensor H24/KV4/D256/B880; native geometry/reducer unchanged')
    if not a.check_only:
        assert not a.archive.exists(),'Never overwrite a prior source archive'
        a.archive.mkdir(parents=True)
        for name,data in original.items():(a.archive/(name+'.py')).write_bytes(data)
        for name in ('unified','env'):(a.vllm_root/FILES[name]).write_bytes(updated[name])
        a.archive.parent.joinpath('manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
