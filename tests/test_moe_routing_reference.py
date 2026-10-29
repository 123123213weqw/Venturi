import pytest
import torch
import torch.nn.functional as F

from venturi.models.moe_routing import RoutingSpec, reference_route_experts, reference_expert_activation


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("top_k", [1,2,6])
def test_router_selects_with_bias_but_uses_unbiased_weights(scoring,top_k):
    x = torch.tensor([[-100.,-3.,-1.,0.,1.,2.,3.,100.]])
    bias = torch.tensor([200., 100., 0., 0., 0., 0., 0., -200.])
    spec = RoutingSpec(top_k,scoring,True,scale=1.5,temperature=2.)
    weights,ids = reference_route_experts(x,bias,spec)
    z=x/2
    scores=z.softmax(-1) if scoring=='softmax' else z.sigmoid() if scoring=='sigmoid' else F.softplus(z).sqrt()
    expected_ids=(scores+bias).topk(top_k,-1)[1]
    expected=scores.gather(1,expected_ids)
    if top_k>1:expected=expected/(expected.sum(-1,keepdim=True)+1e-20)
    assert torch.equal(ids,expected_ids)
    assert torch.equal(weights,expected*1.5)
    assert torch.isfinite(weights).all()


def test_token_specific_bias_and_no_normalization():
    x=torch.zeros(2,3)
    bias=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    weights,ids=reference_route_experts(x,bias,RoutingSpec(1,'sigmoid',False))
    assert ids.tolist()==[[0],[1]] and weights.tolist()==[[.5],[.5]]


def test_clamp_is_asymmetric_on_negative_gate():
    gate=torch.tensor([[-100.,-2.,2.,100.]])
    up=torch.tensor([[-100.,-2.,2.,100.]])
    y=reference_expert_activation(gate,up,limit=10.)
    assert torch.equal(y,F.silu(gate.clamp(max=10))*up.clamp(-10,10))
    assert not torch.equal(y,F.silu(gate.clamp(-10,10))*up.clamp(-10,10))
    assert torch.equal(reference_expert_activation(gate,up,limit=0),F.silu(gate)*up)


@pytest.mark.parametrize("change", [dict(top_k=True),dict(top_k=0),dict(scoring='unknown'),
    dict(temperature=0),dict(scale=float('nan')),dict(normalization_epsilon=-1),dict(scale=True)])
def test_invalid_spec(change):
    with pytest.raises((TypeError,ValueError)):
        RoutingSpec(**(dict(top_k=1,scoring='sigmoid',normalize=True)|change))


def test_bad_shapes_precision_and_nonfinite_fail_closed():
    spec=RoutingSpec(2,'sqrtsoftplus',True)
    for x,b in ((torch.zeros(2,2,dtype=torch.float16),torch.zeros(2)),
                (torch.zeros(2,2),torch.zeros(3)),
                (torch.zeros(2,1),torch.zeros(1)),
                (torch.tensor([[float('nan'),1.]]),torch.zeros(2))):
        with pytest.raises(ValueError):reference_route_experts(x,b,spec)
    with pytest.raises(ValueError):reference_expert_activation(torch.zeros(2,2),torch.zeros(2,3),limit=10)
