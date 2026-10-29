"""Compare CPU component semantics with pinned official pure-Torch methods.

Only Gate.forward and Expert.forward are compiled from the explicitly hashed
reference snapshot. No imports, model initialization, kernels or checkpoint code
are executed. Synthetic inputs: NOT real full-model correctness or performance.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from venturi.models.moe_routing import RoutingSpec, reference_route_experts, reference_expert_activation


def method(source, class_name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
    fn.decorator_list = []
    body = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(torch=torch, F=F, linear=F.linear)
    exec(compile(body, '<hashed official '+class_name+'.forward>', 'exec'), namespace)
    return namespace['forward']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-source', required=True)
    p.add_argument('--reference-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(argv)
    raw = Path(a.reference_source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != a.reference_sha256:
        raise ValueError('Official source SHA256 mismatch')
    gate, expert = method(raw.decode(),'Gate'), method(raw.decode(),'Expert')
    rng = torch.Generator().manual_seed(4171017)
    records=[]
    with torch.no_grad():
        for scoring in ('softmax','sigmoid','sqrtsoftplus'):
            for topk in (1,6):
                for normalize in (False,True):
                    for masked in (False,True):
                        weight = torch.randn(384,32,generator=rng)
                        x = torch.randn(8,32,generator=rng)
                        bias = torch.randn(384,generator=rng)
                        bias_vl = torch.randn(384,generator=rng) if masked else None
                        mask = torch.arange(8)%2==0 if masked else None
                        oracle = SimpleNamespace(weight=weight,bias=bias,bias_vl=bias_vl,
                            topk=topk,score_func=scoring,gate_temp=1.25,
                            norm_topk_prob=normalize,route_scale=1.5)
                        expected,expected_ids=gate(oracle,x,mask)
                        selection_bias=torch.where(mask.unsqueeze(-1),bias_vl,bias) if masked else bias
                        observed,ids=reference_route_experts(F.linear(x,weight),selection_bias,
                            RoutingSpec(topk,scoring,normalize,temperature=1.25,scale=1.5))
                        equal=torch.equal(observed,expected) and torch.equal(ids,expected_ids)
                        records.append(dict(kind='router',scoring=scoring,top_k=topk,
                            normalize=normalize,image_bias=masked,byte_exact=equal,
                            max_abs=(observed-expected).abs().max().item()))
        for limit in (0.,10.):
            for weighted in (False,True):
                g=torch.randn(8,32,generator=rng)*30
                u=torch.randn(8,32,generator=rng)*30
                x=torch.zeros_like(g)
                w=torch.rand(8,1,generator=rng) if weighted else None
                oracle=SimpleNamespace(w1=lambda x:g,w3=lambda x:u,w2=lambda x:x,swiglu_limit=limit)
                expected=expert(oracle,x,w)
                observed=reference_expert_activation(g,u,limit=limit)
                if weighted:observed=w*observed
                records.append(dict(kind='activation',limit=limit,weighted=weighted,
                    byte_exact=torch.equal(observed,expected),max_abs=(observed-expected).abs().max().item()))
    result=dict(scope='Synthetic CPU routing/activation methods only; no full model, weights or TPS',
        official_repository='deepseek-ai/DeepSeek-V4.1-Flash-1017',
        official_revision='1bae0e1088993b9b82de08f59de5be5a34e42170',
        official_source_sha256=digest,torch_version=torch.__version__,device='cpu',
        all_passed=all(r['byte_exact'] for r in records),cases=records)
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if not result['all_passed']:raise SystemExit(1)


if __name__ == '__main__':
    main()
