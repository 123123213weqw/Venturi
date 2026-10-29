import pytest
import torch

from venturi.models.engram_projection import reference_engram_from_rows
from venturi.models.engram_residual import reference_engram_residual


def inputs(dtype=torch.float32):
    g=torch.Generator().manual_seed(417162)
    return [torch.randn(2,3,4,8,generator=g).to(dtype),torch.randn(2,3,3,4,generator=g),
            torch.randn(40,12,generator=g),torch.randn(4,8,generator=g),torch.randn(4,8,generator=g)]


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_explicit_f32_projection_then_gate_matches_independent_split_and_no_alias(dtype):
    tensors=inputs(dtype);stream,rows,w,q,k=tensors
    before=[t.clone() for t in tensors]
    actual=reference_engram_from_rows(*tensors,norm_epsilon=1e-20)
    projection=rows.reshape(6,12)@w.T
    key=projection[:,:32].reshape(2,3,4,8);value=projection[:,32:].reshape(2,3,8)
    expected=reference_engram_residual(stream,key,value,q,k,norm_epsilon=1e-20)
    assert torch.equal(actual,expected) and actual.dtype==dtype
    for tensor,copy in zip(tensors,before):assert torch.equal(tensor,copy)
    actual.fill_(777)
    for tensor,copy in zip(tensors,before):assert torch.equal(tensor,copy)


def test_masked_positions_are_stream_identity_and_no_autograd_graph_is_built():
    tensors=inputs();tensors[1].requires_grad_(True);tensors[2].requires_grad_(True)
    mask=torch.tensor([[True,False,True],[False,True,False]])
    actual=reference_engram_from_rows(*tensors,norm_epsilon=1e-20,token_mask=mask)
    assert torch.equal(actual[~mask],tensors[0][~mask]) and not actual.requires_grad


@pytest.mark.parametrize('index,change',[
    (0,lambda t:t[:,:,0,:]), (1,lambda t:t[:,:2]), (1,lambda t:t.bfloat16()),
    (2,lambda t:t.T), (2,lambda t:t.half()), (3,lambda t:t[0]), (4,lambda t:t.half()),
])
def test_row_head_copy_projection_layout_and_precision_are_not_inferred(index,change):
    tensors=inputs();tensors[index]=change(tensors[index])
    with pytest.raises(ValueError):reference_engram_from_rows(*tensors,norm_epsilon=1e-20)


@pytest.mark.parametrize('index',range(5))
def test_nonfinite_prepared_inputs_rejected_even_if_masked(index):
    tensors=inputs();tensors[index].flatten()[0]=float('nan')
    with pytest.raises(ValueError,match='Finite'):
        reference_engram_from_rows(*tensors,norm_epsilon=1e-20,token_mask=torch.zeros(2,3,dtype=torch.bool))


@pytest.mark.parametrize('budget',[{'max_weight_values':True},{'max_weight_values':479},
    {'max_work_values':239},{'max_work_values':0},{'max_weight_values':2**28+1}])
def test_explicit_weight_and_activation_budgets(budget):
    with pytest.raises(ValueError):reference_engram_from_rows(*inputs(),norm_epsilon=1e-20,**budget)


def test_query_and_key_weights_are_required_not_replaced_by_ones():
    tensors=inputs()
    with pytest.raises(ValueError):reference_engram_from_rows(*tensors[:3],None,tensors[4],norm_epsilon=1e-20)


def test_finite_inputs_with_overflowing_projection_fail_not_plausible_text_state():
    tensors=inputs();tensors[1].fill_(3e38);tensors[2].fill_(3e38)
    with pytest.raises(ValueError):reference_engram_from_rows(*tensors,norm_epsilon=1e-20)
