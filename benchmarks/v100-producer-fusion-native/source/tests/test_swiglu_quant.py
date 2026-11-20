"""Producer fusion preserves the old F16 boundary and both A8 planes."""
import ctypes as C
import os

import pytest
import torch


@pytest.fixture(scope='module')
def sq():
    path,old=os.environ.get('VENTURI_SQ_LIBRARY'),os.environ.get('VENTURI_SQ_CONTROL')
    if not path or not old:pytest.skip('Explicit SM70 SwiGLU fusion and pinned control libraries required')
    p,c=C.CDLL(path),C.CDLL(old)
    signatures={
        'venturi_swiglu_run':[C.c_void_p]*3+[C.c_int64,C.c_int,C.c_void_p],
        'venturi_swiglu_clone_run':[C.c_void_p]*3+[C.c_int64,C.c_int,C.c_void_p],
        'venturi_q8_dp4a_quantize':[C.c_void_p]*2+[C.c_uint64]+[C.c_int]*3+[C.c_void_p],
        'venturi_swiglu_q8_quantize':[C.c_void_p]*3+[C.c_uint64]+[C.c_int]*3+[C.c_void_p]*2,
    }
    for lib in (p,c):
        for name,args in signatures.items():
            if hasattr(lib,name):
                fn=getattr(lib,name);fn.argtypes=args;fn.restype=C.c_int
    return p,c


def ok(code):assert code==0


def baseline(c,gate,up,mid,work):
    t,k=gate.shape;stream=torch.cuda.current_stream().cuda_stream
    ok(c.venturi_swiglu_run(gate.data_ptr(),up.data_ptr(),mid.data_ptr(),t*k,0,stream))
    ok(c.venturi_q8_dp4a_quantize(mid.data_ptr(),work.data_ptr(),work.numel(),t,k,2,stream))


def fused(p,gate,up,work,threads,debug=None):
    t,k=gate.shape
    return p.venturi_swiglu_q8_quantize(gate.data_ptr(),up.data_ptr(),work.data_ptr(),work.numel(),t,k,threads,
        debug.data_ptr() if debug is not None else None,torch.cuda.current_stream().cuda_stream)


def equal_half(actual,expected):
    # NaN payloads are not the public arithmetic contract; encoded A8 scales
    # canonicalize invalid groups and must still match every byte.
    a,b=actual.cpu(),expected.cpu();assert torch.equal(a.isnan(),b.isnan())
    finite_or_inf=~b.isnan()
    assert torch.equal(a[finite_or_inf].contiguous().view(torch.uint8),b[finite_or_inf].contiguous().view(torch.uint8))


@pytest.mark.parametrize('threads',[128,256])
@pytest.mark.parametrize('tokens',[1,2,7,8,23,32])
@pytest.mark.parametrize('k',[1,31,32,33,67,1025,8704,17408])
def test_swiglu_quant_full_bytes_tails_guards_and_fp64(sq,threads,tokens,k):
    p,c=sq;gen=torch.Generator().manual_seed(k+tokens)
    gate=torch.randn(tokens,k,generator=gen).half().cuda()
    up=torch.randn(tokens,k,generator=gen).half().cuda()
    expected=torch.empty_like(gate)
    size=tokens*((k+31)//32)*72
    reference=torch.empty(size,device='cuda',dtype=torch.uint8)
    storage=torch.full((size+32,),47,device='cuda',dtype=torch.uint8);work=storage[16:-16]
    output_guard=torch.full((tokens*k+2,),47,device='cuda',dtype=torch.float16);debug=output_guard[1:-1].view(tokens,k)
    baseline(c,gate,up,expected,reference)
    ok(fused(p,gate,up,work,threads));assert torch.equal(work,reference)
    ok(fused(p,gate,up,work,threads,debug));assert torch.equal(work,reference)
    equal_half(debug,expected)
    assert bool((storage[:16]==47).all()) and bool((storage[-16:]==47).all())
    assert output_guard[0]==output_guard[-1]==47
    g,u=gate.cpu().double(),up.cpu().double()
    independent=(g*torch.sigmoid(g)*u).half()
    torch.testing.assert_close(expected.cpu(),independent,rtol=2*torch.finfo(torch.float16).eps,atol=2e-6)


@pytest.mark.parametrize('threads',[128,256])
@pytest.mark.parametrize('up_value',[0.0,0.5,-1.0,0.0001,65504.0,float('inf')])
def test_exhaustive_f16_gate_patterns_and_extremes(sq,threads,up_value):
    p,c=sq;k=65536
    gate=torch.arange(k,dtype=torch.int32).short().view(torch.float16).reshape(1,k).cuda()
    up=torch.full_like(gate,up_value);mid=torch.empty_like(gate);debug=torch.empty_like(gate)
    reference=torch.empty((k//32)*72,device='cuda',dtype=torch.uint8);work=torch.empty_like(reference)
    baseline(c,gate,up,mid,reference)
    ok(fused(p,gate,up,work,threads));assert torch.equal(work,reference)
    ok(fused(p,gate,up,work,threads,debug));assert torch.equal(work,reference)
    equal_half(debug,mid)


@pytest.mark.parametrize('threads',[128,256])
@pytest.mark.parametrize('debug_alias',['gate','up'])
def test_exact_inplace_debug_alias(sq,threads,debug_alias):
    p,c=sq;t,k=3,67
    gen=torch.Generator().manual_seed(74)
    gate=torch.randn(t,k,generator=gen).half().cuda();up=torch.randn(t,k,generator=gen).half().cuda()
    mid=torch.empty_like(gate);reference=torch.empty(t*((k+31)//32)*72,device='cuda',dtype=torch.uint8);work=torch.empty_like(reference)
    baseline(c,gate,up,mid,reference)
    debug=gate if debug_alias=='gate' else up
    ok(fused(p,gate,up,work,threads,debug));assert torch.equal(work,reference)
    equal_half(debug,mid)


@pytest.mark.parametrize('threads',[128,256])
def test_live_graph_nonfinite_zero_and_changed_inputs(sq,threads):
    p,c=sq;t,k=9,8704
    gate=torch.empty(t,k,device='cuda',dtype=torch.float16);up=torch.empty_like(gate);mid=torch.empty_like(gate)
    work=torch.empty(t*(k//32)*72,device='cuda',dtype=torch.uint8);ref=torch.empty_like(work)
    gen=torch.Generator().manual_seed(91);g=torch.randn(t,k,generator=gen).half().cuda();u=torch.randn(t,k,generator=gen).half().cuda()
    gate.copy_(g);up.copy_(u)
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):ok(fused(p,gate,up,work,threads))
    torch.cuda.current_stream().wait_stream(side)
    for mode in ('normal','zero','nonfinite','changed'):
        if mode=='zero':gate.zero_();up.fill_(1)
        elif mode=='nonfinite':gate[0,1]=float('inf');up[0,1]=0
        elif mode=='changed':gate.copy_(-g);up.copy_(u*0.5)
        graph.replay();baseline(c,gate,up,mid,ref);torch.cuda.synchronize()
        assert torch.equal(work,ref)


@pytest.mark.parametrize('bad',['null','short','threads','gate_alignment','work_alignment','input_overlap','debug_work_overlap','debug_partial_overlap'])
def test_invalid_geometry_and_overlap_fail_closed(sq,bad):
    p,_=sq;t,k=1,67
    gate=torch.ones(t,k,device='cuda',dtype=torch.float16);up=torch.ones_like(gate)
    work=torch.empty(((k+31)//32)*72,device='cuda',dtype=torch.uint8)
    args=[gate.data_ptr(),up.data_ptr(),work.data_ptr(),work.numel(),t,k,256,None,torch.cuda.current_stream().cuda_stream]
    if bad=='null':args[0]=None
    elif bad=='short':args[3]-=1
    elif bad=='threads':args[6]=64
    elif bad=='gate_alignment':args[0]+=1
    elif bad=='work_alignment':args[2]+=1
    elif bad=='input_overlap':args[2]=gate.data_ptr()
    elif bad=='debug_work_overlap':args[7]=work.data_ptr()
    elif bad=='debug_partial_overlap':args[7]=gate.data_ptr()+2
    assert p.venturi_swiglu_q8_quantize(*args)!=0
