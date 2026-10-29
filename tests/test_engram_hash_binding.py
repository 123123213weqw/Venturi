import pytest
import torch

from venturi.gguf_metadata import read_verified_integer_metadata
from venturi.models.engram_hash import EngramHashState,bind_engram_hash_metadata
from test_gguf_metadata import write_gguf


NAMES=('engram_token_map.weight','engram_primes.weight','engram_offsets.weight','engram_multipliers.weight')
LAYOUT=dict(layer_ids=(1,),max_ngram_size=3,n_heads=2,compressed_vocab_size=3,pad_token_id=2,vocab_size=5,table_rows=(36,))


def tensors():
    return [(NAMES[0],26,(5,),(0,1,2,1,0)),(NAMES[1],27,(1,4),(5,7,11,13)),
        (NAMES[2],27,(1,4),(0,5,12,23)),(NAMES[3],27,(1,3),(3,5,7))]


def load(root,items):
    asset=write_gguf(root,items)
    return read_verified_integer_metadata(asset,root,NAMES)


def test_verified_integer_rows_bind_to_spec_and_real_hash_equations(tmp_path):
    metadata=load(tmp_path,tensors());spec=bind_engram_hash_metadata(metadata,**LAYOUT)
    out=EngramHashState(spec,batch_size=1,max_sequence_tokens=4)(torch.tensor([[1]]),0)
    assert out.tolist()==[[[[4,7,19,30]]]]
    assert spec.token_map==metadata.tensor(NAMES[0]).values and spec.table_rows==(36,)


@pytest.mark.parametrize('field,change',[
    (0,(NAMES[0],26,(1,5),(0,1,2,1,0))),
    (1,(NAMES[1],26,(1,4),(5,7,11,13))),
    (2,(NAMES[2],27,(1,4),(0,7,12,23))),
    (3,(NAMES[3],27,(1,3),(3,5,2**63-1))),
    (0,(NAMES[0],26,(5,),(0,-1,2,1,0))),
])
def test_valid_full_sha_does_not_mean_semantically_valid_hash_metadata(tmp_path,field,change):
    data=tensors();data[field]=change;metadata=load(tmp_path,data)
    with pytest.raises(ValueError):bind_engram_hash_metadata(metadata,**LAYOUT)


@pytest.mark.parametrize('change',[{'n_heads':True},{'n_heads':4},{'max_ngram_size':4},
    {'vocab_size':6},{'compressed_vocab_size':4},{'pad_token_id':5},{'table_rows':(37,)},{'table_rows':(36.0,)},{'table_rows':(True,)}])
def test_architecture_layout_mismatch_fails_closed(tmp_path,change):
    metadata=load(tmp_path,tensors())
    with pytest.raises(ValueError):bind_engram_hash_metadata(metadata,**(LAYOUT|change))


def test_missing_metadata_and_unqualified_inventory_are_not_accepted(tmp_path):
    asset=write_gguf(tmp_path,tensors())
    metadata=read_verified_integer_metadata(asset,tmp_path,NAMES[:2])
    with pytest.raises(ValueError,match='Missing required'):bind_engram_hash_metadata(metadata,**LAYOUT)
    with pytest.raises(TypeError):bind_engram_hash_metadata({'header_only':True},**LAYOUT)
