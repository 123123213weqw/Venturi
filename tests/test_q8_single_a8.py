"""Single-pass correctness is distinct from preserving two-pass model quality."""
import ctypes as C

import pytest
import torch

from q8_dp4a_reference import projection_reference
from test_q8_prefetch import api, operands


@pytest.fixture(scope='module')
def single(api):
    m,p,c = api
    p.venturi_q8_a8_fused_run.argtypes = p.venturi_q8_fused_run.argtypes
    p.venturi_q8_a8_fused_run.restype = C.c_int
    p.venturi_q8_a8_partial_run.argtypes = [C.c_void_p,C.c_uint64]+[C.c_void_p]*3+[C.c_int]*5+[C.c_void_p]
    p.venturi_q8_a8_partial_run.restype = C.c_int
    return m,p,c


SHAPES = [(67,[33,17],False),(1025,[1031],True),
          (5120,[8704,8704],False),(8704,[5120],True),
          (5120,[5120,3072],False),(3072,[5120],True),
          (5120,[6144,512,512],False)]


@pytest.mark.parametrize('passes',[1,2])
@pytest.mark.parametrize('tokens',[1,2,4,8,9,23,32])
@pytest.mark.parametrize('k,ns,partial',SHAPES)
def test_single_a8_bytes_fp64_tails_and_guards(single,passes,tokens,k,ns,partial):
    m,p,c = single
    padded,weights,gpu,xcpu,x,_ = operands(k,ns,tokens)
    work = torch.empty(tokens*((k+31)//32)*36*passes,device='cuda',dtype=torch.uint8)
    stream = torch.cuda.current_stream().cuda_stream
    m.check(p.venturi_q8_dp4a_quantize(x.data_ptr(),work.data_ptr(),work.numel(),tokens,k,passes,stream))
    dtype = torch.float32 if partial else torch.float16
    strides = ns if partial else [n+5 for n in ns]
    guards = [torch.full((tokens*s+2,),47,device='cuda',dtype=dtype) for s in strides]
    outputs = [g[1:-1].view(tokens,s) for g,s in zip(guards,strides)]
    if partial:
        args = [work.data_ptr(),work.numel(),gpu[0][0].data_ptr(),gpu[0][1].data_ptr(),outputs[0].data_ptr(),tokens,ns[0],k,padded,passes,stream]
        m.check(p.venturi_q8_a8_partial_run(*args))
    else:
        desc = (m.Descriptor*len(ns))(*[m.Descriptor(q.data_ptr(),s.data_ptr(),y.data_ptr(),n,stride) for (q,s),y,n,stride in zip(gpu,outputs,ns,strides)])
        args = [work.data_ptr(),work.numel(),desc,len(ns),tokens,k,padded,passes,stream]
        m.check(p.venturi_q8_a8_fused_run(*args))
    actual = [y[:,:n].cpu().contiguous() for y,n in zip(outputs,ns)]
    if passes==2 or not partial:
        if partial:
            fn = c.venturi_q8_projection_partial_run
            m.check(fn(*args[:-2],stream))
        else:
            fn = c.venturi_q8_projection_fused_run if passes==2 else c.venturi_q8_fused_run
            m.check(fn(*args))
        assert all(torch.equal(y.view(torch.uint8),v[:,:n].cpu().contiguous().view(torch.uint8)) for y,v,n in zip(actual,outputs,ns))
    for (q,s),y in zip(weights,actual):
        rows = sorted(set([0,len(q)-1]+list(range(min(8,len(q))))))
        ref = projection_reference(xcpu,q[rows],s[rows],passes)
        error = float((y[:,rows].double()-ref).norm()/(ref.norm()+1e-30))
        assert error<(1e-5 if partial else 0.001)
    for g,y,n in zip(guards,outputs,ns):
        assert g[0]==g[-1]==47 and bool((y[:,n:]==47).all())


@pytest.mark.parametrize('passes',[1,2])
@pytest.mark.parametrize('partial',[False,True])
def test_single_graph_changed_nonfinite_zero_and_finite_inputs(single,passes,partial):
    m,p,_ = single
    t,k = 9,67;ns = [33] if partial else [33,17]
    padded,weights,gpu,xcpu,x,_ = operands(k,ns,t)
    work = torch.empty(t*((k+31)//32)*36*passes,device='cuda',dtype=torch.uint8)
    ys = [torch.empty(t,n,device='cuda',dtype=torch.float32 if partial else torch.float16) for n in ns]
    desc = (m.Descriptor*len(ns))(*[m.Descriptor(q.data_ptr(),s.data_ptr(),y.data_ptr(),n,n) for (q,s),y,n in zip(gpu,ys,ns)])
    def call():
        stream = torch.cuda.current_stream().cuda_stream
        m.check(p.venturi_q8_dp4a_quantize(x.data_ptr(),work.data_ptr(),work.numel(),t,k,passes,stream))
        if partial:m.check(p.venturi_q8_a8_partial_run(work.data_ptr(),work.numel(),gpu[0][0].data_ptr(),gpu[0][1].data_ptr(),ys[0].data_ptr(),t,ns[0],k,padded,passes,stream))
        else:m.check(p.venturi_q8_a8_fused_run(work.data_ptr(),work.numel(),desc,len(ns),t,k,padded,passes,stream))
    side = torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):call()
    torch.cuda.current_stream().wait_stream(side)
    x.fill_(float('inf'));graph.replay();torch.cuda.synchronize()
    assert all(bool(y.isnan().all()) for y in ys)
    x.zero_();graph.replay();torch.cuda.synchronize()
    assert all(not bool(y.any()) for y in ys)
    x.copy_(xcpu);graph.replay();torch.cuda.synchronize()
    for (q,s),y in zip(weights,ys):
        ref = projection_reference(xcpu,q,s,passes)
        assert float((y.cpu().double()-ref).norm()/(ref.norm()+1e-30))<0.001


def test_lost_small_feature_is_a_quality_counterexample_not_a_kernel_bug(single):
    m,p,_ = single
    x = torch.zeros(1,32,device='cuda',dtype=torch.float16);x[0,0]=1;x[0,1]=0.003
    codes = torch.zeros(1,128,device='cuda',dtype=torch.int8);codes[0,1]=127
    scale = torch.ones(1,4,device='cuda',dtype=torch.float16)
    results = []
    for passes in (1,2):
        work = torch.empty(36*passes,device='cuda',dtype=torch.uint8)
        y = torch.empty(1,1,device='cuda',dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        m.check(p.venturi_q8_dp4a_quantize(x.data_ptr(),work.data_ptr(),work.numel(),1,32,passes,stream))
        m.check(p.venturi_q8_a8_partial_run(work.data_ptr(),work.numel(),codes.data_ptr(),scale.data_ptr(),y.data_ptr(),1,1,32,128,passes,stream))
        results.append(float(y.item()))
    ideal = float(x[0,1].item())*127
    assert results[0]==0 and abs(results[1]-ideal)/ideal<1e-5


@pytest.mark.parametrize('passes',[-1,0,3])
def test_single_invalid_passes_fail_closed(single,passes):
    _,p,_ = single
    assert p.venturi_q8_a8_fused_run(None,0,None,2,8,5120,5120,passes,None)!=0
    assert p.venturi_q8_a8_partial_run(None,0,None,None,None,8,5120,8704,8704,passes,None)!=0
