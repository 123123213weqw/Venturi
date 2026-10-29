"""Selected official Engram gate/residual method vs CPU oracle, not full Engram.

Embedding and wkv projection are replaced by already prepared synthetic key/value
inputs. Hashing, real weights, table dequantization, GPU and TPS are NOT checked.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from venturi.models.engram_residual import reference_engram_residual


def qualify(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engram')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    fn.decorator_list=[]
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<hashed official Engram.forward>','exec'),scope)
    rng=torch.Generator().manual_seed(417141)
    records=[]
    with torch.no_grad():
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            for b,t,d in ((1,1,8),(2,3,32),(1,2,5120)):
                x=torch.randn(b,t,4,d,generator=rng).to(dtype)
                key=torch.randn(b,t,4,d,generator=rng).to(dtype)
                value=torch.randn(b,t,d,generator=rng).to(dtype)
                q=torch.randn(4,d,generator=rng);k=torch.randn(4,d,generator=rng)
                projected=torch.cat((key.flatten(-2),value),dim=-1)
                for eps in (1e-20,1e-6):
                    fake=SimpleNamespace(hc_mult=4,dim=d,eps=eps,clamp_value=1e-6,q_weight=q,k_weight=k,
                        embed=lambda ids:torch.zeros(*ids.shape,1),wkv=lambda unused:projected)
                    for mask in (None,torch.zeros(b,t,dtype=torch.bool),torch.arange(b*t).reshape(b,t)%2==0):
                        actual=reference_engram_residual(x,key,value,q,k,norm_epsilon=eps,token_mask=mask)
                        expected=scope['forward'](fake,x,torch.zeros(b,t,24,dtype=torch.long),mask)
                        records.append(dict(dtype=str(dtype),shape=list(x.shape),norm_epsilon=eps,
                            mask='none' if mask is None else mask.tolist(),byte_exact=torch.equal(actual,expected),
                            max_abs=(actual.float()-expected.float()).abs().max().item()))
    return records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-source',required=True)
    p.add_argument('--reference-sha256',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    source=Path(a.reference_source).read_bytes();digest=hashlib.sha256(source).hexdigest()
    if digest!=a.reference_sha256:raise ValueError('Official source SHA256 mismatch')
    cases=qualify(source.decode())
    result=dict(scope='Synthetic prepared-key/value CPU gate/residual only; NOT hashing, embedding, projection, weights, GPU or model correctness/TPS',
        official_repository='deepseek-ai/DeepSeek-V4.1-Flash-1017',official_revision='1bae0e1088993b9b82de08f59de5be5a34e42170',
        official_source_sha256=digest,torch_version=torch.__version__,device='cpu',cases=cases,
        all_byte_exact=all(r['byte_exact'] for r in cases))
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if not result['all_byte_exact']:raise SystemExit(1)


if __name__=='__main__':main()
