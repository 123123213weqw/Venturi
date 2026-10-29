"""Predeclared cross-device FP16 gates; failures are evidence, not auto-tuned away."""
from dataclasses import fields,is_dataclass
import torch

GATES=dict(component=dict(atol=.002,peak_rtol=.003,relative_l2=.003),
           logits=dict(atol=.05,peak_rtol=0.,relative_l2=.005),
           state=dict(atol=.005,peak_rtol=.005,relative_l2=.005))


def difference(actual,expected,*,kind):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        raise AssertionError('Shape/dtype differs across expert policies')
    a,b=actual.detach().cpu(),expected.detach().cpu()
    if not a.is_floating_point():
        return dict(exact=torch.equal(a,b),passed=torch.equal(a,b),shape=list(a.shape),dtype=str(a.dtype))
    a,b=a.float(),b.float()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:return dict(passed=False,finite=False)
    delta=a-b;maximum=float(delta.abs().max()) if a.numel() else 0.;scale=float(b.abs().max()) if a.numel() else 0.
    norm=float(torch.linalg.vector_norm(b));error=float(torch.linalg.vector_norm(delta));relative=error/max(norm,1e-12)
    gate=GATES[kind];passed=maximum<=gate['atol']+gate['peak_rtol']*scale and (relative<=gate['relative_l2'] or norm<1e-4 and maximum<=gate['atol'])
    cosine=float(torch.nn.functional.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))) if a.numel() and norm>1e-8 else None
    return dict(passed=passed,exact=torch.equal(a,b),max_abs=maximum,reference_peak=scale,relative_l2=relative,cosine=cosine,finite=finite,shape=list(a.shape),dtype=str(actual.dtype))


def state_to_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if is_dataclass(value):return type(value)(**{f.name:state_to_cpu(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,tuple):return tuple(state_to_cpu(x) for x in value)
    if isinstance(value,list):return [state_to_cpu(x) for x in value]
    if isinstance(value,dict):return {k:state_to_cpu(v) for k,v in value.items()}
    return value


def compare_state(a,b,path='state'):
    if isinstance(a,torch.Tensor):return [dict(path=path,**difference(a,b,kind='state'))]
    if is_dataclass(a):return [row for f in fields(a) for row in compare_state(getattr(a,f.name),getattr(b,f.name),path+'.'+f.name)]
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        return [row for i,(x,y) in enumerate(zip(a,b)) for row in compare_state(x,y,path+'.'+str(i))]
    if isinstance(a,dict):
        assert set(a)==set(b)
        return [row for key in a for row in compare_state(a[key],b[key],path+'.'+str(key))]
    return [dict(path=path,exact=a==b,passed=a==b)]
