"""CPU contracts only; actual SM70 graph/model gates run separately."""
import pytest
import torch
from venturi.models.flash_next_q4 import quantize_q4_rtn,decode_q4_fp16
from venturi.models.flash_next_q4_graph import PairLayout,one_route,CoalescedQ4

@pytest.mark.parametrize('shape',[(8,6),(128,64),(131,129),(2560,640)])
def test_coalesced_pair_roundtrip_exact_owned(shape):
 h,i=shape;torch.manual_seed(72);pair=(quantize_q4_rtn(torch.randn(2*i,h).half()),quantize_q4_rtn(torch.randn(h,i).half()));layout=PairLayout(h,i)
 payload=layout.pack_cpu(pair);before=payload.clone();other=layout.unpack(payload)
 assert payload.numel()==sum(w.nbytes for w in pair)
 for x,y in zip(pair,other):
  assert torch.equal(x.codes,y.codes) and torch.equal(x.scales,y.scales)
  assert torch.equal(decode_q4_fp16(x),decode_q4_fp16(y))
 pair[0].codes.zero_();pair[0].scales.zero_();assert torch.equal(before,payload)
 assert all(w.codes.untyped_storage().data_ptr()==payload.untyped_storage().data_ptr() for w in other)

@pytest.mark.parametrize('shape',[(0,3),(True,6),(8,-1),(32769,1),(32768,32768)])
def test_bad_geometry(shape):
 with pytest.raises(ValueError):PairLayout(*shape)

def test_bad_payload_and_scales():
 layout=PairLayout(8,6);pair=(quantize_q4_rtn(torch.ones(12,8)),quantize_q4_rtn(torch.ones(8,6)))
 for payload in [torch.zeros(layout.nbytes+1,dtype=torch.uint8),torch.zeros(layout.nbytes),torch.zeros(layout.nbytes*2,dtype=torch.uint8)[::2]]:
  with pytest.raises(ValueError):layout.unpack(payload)
 for value in [float('nan'),-1.]:
  pair[0].scales[0,0]=value
  with pytest.raises(ValueError):layout.pack_cpu(pair)
 for bad in [(),pair[:1],('a','b')]:
  with pytest.raises(ValueError):layout.pack_cpu(bad)

def test_route_order_duplicates_and_bounds():
 assert one_route(torch.tensor([[3,1,7,0]]),8)==((0,3),(1,1),(3,0),(7,2))
 assert one_route(torch.tensor([[3,1,3,0]]),8) is None
 for bad in [torch.tensor([[8]]),torch.tensor([[-1]]),torch.tensor([[1.]]),torch.tensor([1]),torch.tensor([[1],[2]])]:
  with pytest.raises(ValueError):one_route(bad,8)

def test_prepare_pair_rejects_scratch_budget_before_cuda():
 native=CoalescedQ4.__new__(CoalescedQ4);native.max_output_bytes=1
 pair=(quantize_q4_rtn(torch.ones(12,8)),quantize_q4_rtn(torch.ones(8,6)))
 with pytest.raises(ValueError,match='scratch'):native.prepare_pair(pair,PairLayout(8,6),device='cuda:0')
