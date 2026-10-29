import math

import pytest
import torch

from venturi.models.engram_residual import reference_engram_residual


def data(dtype=torch.float32):
    x=torch.tensor([[[[1.,2.],[-3.,4.]]]],dtype=dtype)
    key=torch.tensor([[[[2.,-3.],[4.,5.]]]],dtype=dtype)
    value=torch.tensor([[[.25,-.5]]],dtype=dtype)
    q=torch.tensor([[1.,2.],[3.,4.]])
    k=torch.tensor([[2.,3.],[4.,5.]])
    return x,key,value,q,k


def test_per_copy_normalized_gate_matches_independent_scalar_equations():
    tensors=data();x,key,value,q,k=tensors;eps=1e-20
    expected=[]
    for h in range(2):
        xs,ks=x[0,0,h].tolist(),key[0,0,h].tolist()
        a=sum(t*t for t in xs)/2+eps;b=sum(t*t for t in ks)/2+eps
        dot=sum(xs[i]*float(q[h,i])*float(k[h,i])*ks[i] for i in range(2))/math.sqrt(a)/math.sqrt(b)/math.sqrt(2)
        signed=math.copysign(math.sqrt(max(abs(dot),1e-6)),dot)
        gate=1/(1+math.exp(-signed))
        expected.append([xs[i]+gate*float(value[0,0,i]) for i in range(2)])
    got=reference_engram_residual(*tensors,norm_epsilon=eps)
    torch.testing.assert_close(got,torch.tensor([[expected]]),rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_masked_tokens_pass_through_bit_exact_and_stream_dtype_preserved(dtype):
    tensors=data(dtype)
    out=reference_engram_residual(*tensors,norm_epsilon=1e-20,token_mask=torch.zeros(1,1,dtype=torch.bool))
    assert out.dtype==dtype and torch.equal(out,tensors[0])


def test_zero_dot_has_floored_sqrt_before_sigmoid_not_exact_half_gate():
    x=torch.zeros(1,1,1,2);key=torch.zeros_like(x);value=torch.ones(1,1,2)*2;w=torch.ones(1,2)
    out=reference_engram_residual(x,key,value,w,w,norm_epsilon=1e-20)
    expected=2/(1+math.exp(-.001))
    torch.testing.assert_close(out,torch.full_like(out,expected))
    assert bool((out>1).all())


def test_joint_across_stream_normalization_would_be_wrong():
    x,key,value,q,k=data();got=reference_engram_residual(x,key,value,q,k,norm_epsilon=1e-20)
    h=x.float();rstd=torch.rsqrt(h.square().mean((-1,-2))+1e-20)*torch.rsqrt(key.square().mean((-1,-2))+1e-20)
    dot=(h*q*k*key).sum(-1)*rstd.unsqueeze(-1)*2**-.5
    wrong=h+torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(),dot)).unsqueeze(-1)*value.unsqueeze(-2)
    assert not torch.allclose(got,wrong)


@pytest.mark.parametrize('change',[{'norm_epsilon':0},{'norm_epsilon':True},
    {'norm_epsilon':float('nan')},{'signed_sqrt_floor':-1}])
def test_scalar_parameters_fail_closed(change):
    with pytest.raises(ValueError):reference_engram_residual(*data(),**({'norm_epsilon':1e-20}|change))


@pytest.mark.parametrize('field',[0,1,2,3,4])
def test_bad_shapes_fail_before_arithmetic(field):
    tensors=list(data());tensors[field]=tensors[field].flatten()
    with pytest.raises(ValueError,match='shapes'):reference_engram_residual(*tensors,norm_epsilon=1e-20)


def test_invalid_mask_and_nonfinite_inputs():
    with pytest.raises(ValueError,match='mask'):
        reference_engram_residual(*data(),norm_epsilon=1e-20,token_mask=torch.ones(1,1))
    tensors=list(data());tensors[2].fill_(float('nan'))
    with pytest.raises(ValueError,match='Finite'):reference_engram_residual(*tensors,norm_epsilon=1e-20)
