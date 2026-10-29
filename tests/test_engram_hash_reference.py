from dataclasses import replace

import pytest
import torch

from venturi.models.engram_hash import EngramHashSpec,EngramHashState


def spec():
    return EngramHashSpec((0,1,2,1,0),3,2,(1,),3,(((5,7),(11,13)),),((3,5,7),))


def state(batch=1):return EngramHashState(spec(),batch_size=batch,max_sequence_tokens=16)


def test_hand_hash_order_offsets_and_compressed_pad():
    out=state()(torch.tensor([[1]]),0)
    assert out.shape==(1,1,1,4)
    # 2gram: 1*3 xor 2*5 = 9; 3gram: 9 xor 2*7 = 7.
    assert out.tolist()==[[[[4,7,19,30]]]]
    assert spec().offsets==((0,5,12,23),) and spec().table_rows==(36,)


def test_chunked_and_monolithic_and_compressed_token_aliases_match():
    ids=torch.tensor([[1,2,3,4,2,1],[0,3,2,1,4,3]])
    mask=torch.tensor([[1,1,0,1,1,1],[1,0,0,1,1,1]],dtype=torch.bool)
    full=state(2)(ids,0,mask);s=state(2)
    chunks=[s(ids[:,:2],0,mask[:,:2]),s(ids[:,2:5],2,mask[:,2:5]),s(ids[:,5:],5,mask[:,5:])]
    assert torch.equal(torch.cat(chunks,1),full)
    assert torch.equal(state()(torch.tensor([[1,1]]),0),state()(torch.tensor([[3,3]]),0))
    assert all(len(h)==2 for h in s._history)


def test_dead_token_blocks_all_earlier_lookback():
    masked=state()(torch.tensor([[1,1,4]]),0,torch.tensor([[True,False,True]]))
    fresh=state()(torch.tensor([[4]]),0)
    assert torch.equal(masked[:,-1:],fresh)


def test_forks_are_independent_and_reset_is_explicit():
    original=state();original(torch.tensor([[1,2]]),0);child=original.fork()
    assert child.spec is original.spec
    a=original(torch.tensor([[1]]),2);b=child(torch.tensor([[0]]),2)
    assert not torch.equal(a,b)
    assert original.position==child.position==3
    child.reset();assert child.position==0 and child._history==[[]] and original.position==3


@pytest.mark.parametrize('ids,pos,mask',[(torch.tensor([[5]]),0,None),(torch.tensor([[-1]]),0,None),
    (torch.tensor([[1.]]),0,None),(torch.tensor([[1],[2]]),0,None),
    (torch.tensor([[1]]),1,None),(torch.tensor([[1]]),True,None),
    (torch.tensor([[1]]),0,torch.ones(1,1))])
def test_bad_admission_does_not_advance_state(ids,pos,mask):
    s=state()
    with pytest.raises(ValueError):s(ids,pos,mask)
    assert s.position==0 and s._history==[[]]


@pytest.mark.parametrize('change',[
    {'compressed_vocab_size':4},{'pad_token_id':5},{'pad_token_id':True},
    {'layer_ids':(1,1)},{'max_ngram_size':1},{'max_ngram_size':True},
    {'moduli':(((5,),),)},{'moduli':(((5,7),(11,)),)},
    {'moduli':(((5,7),(11,2**63)),)}, {'multipliers':((3,4,7),)},
    {'multipliers':((3,5,2**63-1),)}, {'token_map':(0,2,2,2,0)},
])
def test_invalid_or_overflowing_explicit_hash_schema_refused(change):
    with pytest.raises(ValueError):replace(spec(),**change)


def test_context_capacity_is_not_implicit_rewind_or_eviction():
    s=EngramHashState(spec(),batch_size=1,max_sequence_tokens=1)
    s(torch.tensor([[1]]),0)
    with pytest.raises(ValueError):s(torch.tensor([[1]]),1)
    with pytest.raises(ValueError):s(torch.tensor([[1]]),0)
    assert s.position==1
