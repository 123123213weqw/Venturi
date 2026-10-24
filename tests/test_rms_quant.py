"""Fused RMS statistics must keep the old reduction and F16 rounding boundary."""
import ctypes as C
import os

import pytest
import torch
from test_swiglu_quant import equal_half


@pytest.fixture(scope='module')
def rq():
    a,b=os.environ.get('VENTURI_RMS_QUANT_LIBRARY'),os.environ.get('VENTURI_RMS_QUANT_CONTROL')
    if not a or not b:pytest.skip('Explicit V100 RMS fusion and pinned baseline libraries required')
    p,c=C.CDLL(a),C.CDLL(b)
    p.venturi_rms_q8_quantize.argtypes=[C.c_void_p]*3+[C.c_uint64,C.c_int,C.c_int,C.c_float,C.c_int,C.c_void_p,C.c_void_p];p.venturi_rms_q8_quantize.restype=C.c_int
    c.venturi_rms_run.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_float,C.c_void_p];c.venturi_rms_run.restype=C.c_int
    c.venturi_q8_dp4a_quantize.argtypes=[C.c_void_p]*2+[C.c_uint64]+[C.c_int]*3+[C.c_void_p];c.venturi_q8_dp4a_quantize.restype=C.c_int
    return p,c


def baseline(c,x,w,y,work,eps=1e-6):
    t,k=x.shape;s=torch.cuda.current_stream().cuda_stream
    assert c.venturi_rms_run(x.data_ptr(),w.data_ptr(),y.data_ptr(),t,k,0,eps,s)==0
    assert c.venturi_q8_dp4a_quantize(y.data_ptr(),work.data_ptr(),work.numel(),t,k,2,s)==0


def fused(p,x,w,work,chunk,debug=None,eps=1e-6):
    t,k=x.shape
    return p.venturi_rms_q8_quantize(x.data_ptr(),w.data_ptr(),work.data_ptr(),work.numel(),t,k,eps,chunk,
        debug.data_ptr() if debug is not None else None,torch.cuda.current_stream().cuda_stream)


@pytest.mark.parametrize('chunk',[8,32,128])
@pytest.mark.parametrize('tokens',[1,8,23])
@pytest.mark.parametrize('k',[1,31,33,67,1025,5120,8704])
def test_rms_full_workspace_bits_and_guards(rq,chunk,tokens,k):
    p,c=rq;g=torch.Generator().manual_seed(17+tokens+k)
    x=torch.randn(tokens,k,generator=g).half().cuda();w=(1+0.03*torch.randn(k,generator=g)).cuda()
    y=torch.empty_like(x);debug=torch.empty_like(x);ref=torch.empty(tokens*((k+31)//32)*72,device='cuda',dtype=torch.uint8)
    storage=torch.full((ref.numel()+32,),71,device='cuda',dtype=torch.uint8);work=storage[16:-16]
    baseline(c,x,w,y,ref)
    assert fused(p,x,w,work,chunk)==0;assert torch.equal(work,ref)
    assert fused(p,x,w,work,chunk,debug)==0;assert torch.equal(work,ref);equal_half(debug,y)
    assert bool((storage[:16]==71).all()) and bool((storage[-16:]==71).all())
    xd=x.cpu().double();ideal=(xd*torch.rsqrt(xd.square().mean(-1,keepdim=True)+1e-6)*w.cpu().double()).half()
    torch.testing.assert_close(y.cpu(),ideal,rtol=0.002,atol=2e-6)


@pytest.mark.parametrize('chunk',[8,32,128])
def test_rms_graph_zero_extreme_nonfinite_and_changed_input(rq,chunk):
    p,c=rq;t,k=8,5120
    x=torch.ones(t,k,device='cuda',dtype=torch.float16);w=torch.ones(k,device='cuda');y=torch.empty_like(x)
    ref=torch.empty(t*(k//32)*72,device='cuda',dtype=torch.uint8);work=torch.empty_like(ref)
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=side):assert fused(p,x,w,work,chunk)==0
    torch.cuda.current_stream().wait_stream(side)
    for mode in ['normal','zero','extreme','nonfinite','changed']:
        if mode=='zero':x.zero_()
        if mode=='extreme':x.fill_(65504);x[0,:32]=1e-5
        if mode=='nonfinite':x[0,0]=float('inf');w[1]=float('nan')
        if mode=='changed':x.fill_(-0.23);w.fill_(1.01)
        graph.replay();baseline(c,x,w,y,ref);torch.cuda.synchronize();assert torch.equal(work,ref)


@pytest.mark.parametrize('bad',['short','epsilon','chunk','alias','alignment'])
def test_rms_invalid_operands(rq,bad):
    p,_=rq;x=torch.ones(1,5120,device='cuda',dtype=torch.float16);w=torch.ones(5120,device='cuda')
    work=torch.empty(160*72,device='cuda',dtype=torch.uint8)
    a=[x.data_ptr(),w.data_ptr(),work.data_ptr(),work.numel(),1,5120,1e-6,8,None,torch.cuda.current_stream().cuda_stream]
    if bad=='short':a[3]-=1
    if bad=='epsilon':a[6]=0
    if bad=='chunk':a[7]=16
    if bad=='alias':a[8]=x.data_ptr()
    if bad=='alignment':a[2]+=1
    assert p.venturi_rms_q8_quantize(*a)!=0
