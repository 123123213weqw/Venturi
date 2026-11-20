"""Research prefetch schedules preserve the qualified residual projection bytes."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from q8_dp4a_reference import projection_reference


@pytest.fixture(scope="module")
def api():
    probe = os.environ.get("VENTURI_Q8_PREFETCH_LIBRARY")
    control = os.environ.get("VENTURI_Q8_PREFETCH_CONTROL")
    if not probe or not control:
        pytest.skip("Explicit prefetch research and qualified SM70 libraries required")
    path = Path(__file__).parents[1] / "tools/projection/benchmark_q8_prefetch.py"
    spec = importlib.util.spec_from_file_location("prefetch_probe_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.bind(probe), module.bind(control)


def operands(k, ns, tokens):
    gen = torch.Generator().manual_seed(k + tokens)
    padded = (k + 127) // 128 * 128
    weights = [(
        torch.randint(-128, 128, (n,padded), dtype=torch.int8, generator=gen),
        (torch.rand(n,padded//32,generator=gen)*0.02).half(),
    ) for n in ns]
    gpu = [(q.cuda(),s.cuda()) for q,s in weights]
    xcpu = torch.randn(tokens,k,generator=gen).half()
    x = xcpu.cuda()
    work = torch.empty(tokens*((k+31)//32)*72,device="cuda",dtype=torch.uint8)
    return padded, weights, gpu, xcpu, x, work


@pytest.mark.parametrize("tokens", [1,2,7,8,9,23,32])
@pytest.mark.parametrize("k", [67,1025,5120])
@pytest.mark.parametrize("partial", [False,True])
def test_prefetch_full_bytes_tails_guards_and_fp64(api,tokens,k,partial):
    m,p,c = api
    ns = [1031] if partial else [1025,17,33]
    padded,weights,gpu,xcpu,x,work = operands(k,ns,tokens)
    stream = torch.cuda.current_stream().cuda_stream
    m.check(p.venturi_q8_dp4a_quantize(x.data_ptr(),work.data_ptr(),work.numel(),tokens,k,2,stream))
    reference = None
    for mode in (0,1,2):
        dtype = torch.float32 if partial else torch.float16
        strides = ns if partial else [n+5 for n in ns]
        guards = [torch.full((tokens*s+2,),47,device="cuda",dtype=dtype) for s in strides]
        outputs = [g[1:-1].view(tokens,s) for g,s in zip(guards,strides)]
        if partial:
            args = [work.data_ptr(),work.numel(),gpu[0][0].data_ptr(),gpu[0][1].data_ptr(),outputs[0].data_ptr(),tokens,ns[0],k,padded]
            fn = c.venturi_q8_column_partial_run if mode==0 else p.venturi_q8_prefetch_partial_run
            if mode:
                args.append(mode)
        else:
            desc = (m.Descriptor*len(ns))(*[m.Descriptor(q.data_ptr(),s.data_ptr(),y.data_ptr(),n,stride) for (q,s),y,n,stride in zip(gpu,outputs,ns,strides)])
            args = [work.data_ptr(),work.numel(),desc,len(ns),tokens,k,padded,2 if mode==0 else mode]
            fn = c.venturi_q8_fused_run if mode==0 else p.venturi_q8_prefetch_fused_run
        m.check(fn(*args,stream))
        values = [y[:,:n].cpu().contiguous() for y,n in zip(outputs,ns)]
        if reference is None:
            reference = values
            for (q,s),y in zip(weights,values):
                indices = sorted(set([0,len(q)-1]+list(range(min(8,len(q))))))
                ideal = projection_reference(xcpu,q[indices],s[indices],2)
                assert float((y[:,indices].double()-ideal).norm()/(ideal.norm()+1e-12)) < 0.001
        else:
            assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(reference,values))
        for g,y,n in zip(guards,outputs,ns):
            assert g[0]==g[-1]==47
            assert bool((y[:,n:]==47).all())


@pytest.mark.parametrize("mode", [1,2])
@pytest.mark.parametrize("partial", [False,True])
def test_capture_nonfinite_zero_and_restored_input(api,mode,partial):
    m,p,c = api
    tokens,k = 9,1025
    ns = [33] if partial else [17,33]
    padded,_,gpu,_,x,work = operands(k,ns,tokens)
    original = x.clone()
    dtype = torch.float32 if partial else torch.float16
    outputs = [torch.empty(tokens,n,device="cuda",dtype=dtype) for n in ns]
    expected = [torch.empty_like(y) for y in outputs]
    side = torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())

    def call(control=False):
        stream = torch.cuda.current_stream().cuda_stream
        m.check(p.venturi_q8_dp4a_quantize(x.data_ptr(),work.data_ptr(),work.numel(),tokens,k,2,stream))
        ys = expected if control else outputs
        if partial:
            args = [work.data_ptr(),work.numel(),gpu[0][0].data_ptr(),gpu[0][1].data_ptr(),ys[0].data_ptr(),tokens,ns[0],k,padded]
            fn = c.venturi_q8_column_partial_run if control else p.venturi_q8_prefetch_partial_run
            if not control:args.append(mode)
        else:
            desc = (m.Descriptor*len(ns))(*[m.Descriptor(q.data_ptr(),s.data_ptr(),y.data_ptr(),n,n) for (q,s),y,n in zip(gpu,ys,ns)])
            args = [work.data_ptr(),work.numel(),desc,len(ns),tokens,k,padded,2 if control else mode]
            fn = c.venturi_q8_fused_run if control else p.venturi_q8_prefetch_fused_run
        m.check(fn(*args,stream))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):call()
    torch.cuda.current_stream().wait_stream(side)
    x.fill_(float("inf"));graph.replay();torch.cuda.synchronize()
    assert all(bool(torch.isnan(y).all()) for y in outputs)
    x.zero_();graph.replay();torch.cuda.synchronize()
    assert all(not bool(y.any()) for y in outputs)
    x.copy_(original);graph.replay();call(control=True);torch.cuda.synchronize()
    assert all(torch.equal(y.view(torch.uint8),e.view(torch.uint8)) for y,e in zip(outputs,expected))


def test_raw_bad_geometry_fails_closed(api):
    m,p,_ = api
    assert p.venturi_q8_prefetch_fused_run(None,0,None,2,8,5120,5120,1,None) != 0
    assert p.venturi_q8_prefetch_partial_run(None,0,None,None,None,8,5120,8704,8704,1,None) != 0
