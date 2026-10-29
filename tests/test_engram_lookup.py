"""Generated tiny bound hash parameters + packed tables, NOT real model weights."""
from dataclasses import replace

import pytest
import torch

from venturi.models.engram_bindings import load_deepseek41_engram_hash
from venturi.models.engram_hash import EngramHashState
from venturi.models.engram_lookup import EngramIQ4NLLookup
from test_engram_component_binding import inputs, config, header, REV
from test_gguf_rows import fixture


def components(root, *, lookup_metadata=None, tables=None, head_dim=32):
    cfg=config();cfg['text_config']['engram_head_dim']=head_dim
    hdr=[(k,t,head_dim if k=='deepseek41.engram_head_dim' else v) for k,t,v in header()]
    bundle,kwargs=inputs(root,cfg=cfg,hdr=hdr)
    binding=load_deepseek41_engram_hash(bundle,**kwargs)
    if tables is None:tables=[('blk.1.engram_embd.weight',36,head_dim,20),('blk.3.engram_embd.weight',88,head_dim,20)]
    if lookup_metadata is None:lookup_metadata=[('general.architecture','deepseek41-engram'),('deepseek41.engram.source',REV)]
    asset,expected=fixture(root,tables=tables,metadata=lookup_metadata)
    bundle=replace(bundle,files=bundle.files+(asset,))
    return bundle,binding,asset,expected


def test_full_sha_bound_hash_to_packed_rows_and_merged_request_scatter(tmp_path):
    bundle,binding,asset,expected=components(tmp_path)
    original=EngramHashState(binding.hash_spec,batch_size=1,max_sequence_tokens=8)
    preview=original.fork();hashes=preview(torch.tensor([[1,2,4]]),0);copy=hashes.clone()
    with EngramIQ4NLLookup(bundle,binding,asset,tmp_path) as lookup:
        assert not lookup.model_supported
        for index,layer in enumerate(binding.hash_spec.layer_ids):
            a,b=lookup.lookup_layer_many(layer,[hashes,hashes.clone()])
            assert a.shape==(1,3,4,32)
            reference=expected[f'blk.{layer}.engram_embd.weight'][hashes[:,:,index,:]]
            assert torch.equal(a,reference) and torch.equal(b,reference)
            assert lookup.last_read.requested_rows==24 and lookup.last_read.unique_rows<24
            a.fill_(999)
            assert torch.equal(b,reference)
    assert original.position==0 and preview.position==3 and torch.equal(hashes,copy)
    assert lookup.closed


def test_prefill_decode_chunks_masks_and_batch_two_match_one_shot_rows(tmp_path):
    bundle,binding,asset,_=components(tmp_path)
    state=EngramHashState(binding.hash_spec,batch_size=2,max_sequence_tokens=6)
    tokens=torch.tensor([[1,2,4,0,7,3],[3,8,2,4,6,1]])
    mask=torch.ones_like(tokens,dtype=torch.bool);mask[0,2:4]=False
    full_hashes=state(tokens,0,mask);state.reset()
    with EngramIQ4NLLookup(bundle,binding,asset,tmp_path) as lookup:
        full=lookup.lookup_layer(3,full_hashes)
        parts=[]
        for start,end in ((0,2),(2,3),(3,6)):
            chunk=state(tokens[:,start:end],start,mask[:,start:end])
            parts.append(lookup.lookup_layer(3,chunk))
        assert torch.equal(full,torch.cat(parts,dim=1))


@pytest.mark.parametrize('metadata',[
    [('general.architecture','deepseek41-engram'),('deepseek41.engram.source','b'*40)],
    [('general.architecture','qwen4_exp'),('deepseek41.engram.source',REV)],
    [('general.architecture','deepseek41-engram')],
])
def test_valid_table_sha_is_not_enough_without_matching_source_profile(tmp_path,metadata):
    bundle,binding,asset,_=components(tmp_path,lookup_metadata=metadata)
    with pytest.raises(ValueError,match='scalar identity'):
        EngramIQ4NLLookup(bundle,binding,asset,tmp_path)


@pytest.mark.parametrize('tables',[
    [('blk.1.engram_embd.weight',36,32,2),('blk.3.engram_embd.weight',88,32,20)],
    [('blk.1.engram_embd.weight',37,32,20),('blk.3.engram_embd.weight',88,32,20)],
    [('blk.1.engram_embd.weight',36,64,20),('blk.3.engram_embd.weight',88,32,20)],
    [('blk.1.engram_embd.weight',36,32,20)],
])
def test_bound_table_names_rows_head_width_and_dtype_are_not_inferred_or_repaired(tmp_path,tables):
    bundle,binding,asset,_=components(tmp_path,tables=tables)
    with pytest.raises(ValueError,match='dtype/shape'):
        EngramIQ4NLLookup(bundle,binding,asset,tmp_path)


def test_bundle_membership_role_and_component_revision_must_agree(tmp_path):
    bundle,binding,asset,_=components(tmp_path)
    for changed in (replace(bundle,official_revision='b'*40),replace(bundle,files=(asset,))):
        with pytest.raises(ValueError,match='source-bound'):EngramIQ4NLLookup(changed,binding,asset,tmp_path)
    role=replace(asset,role='backbone')
    other=replace(bundle,files=(binding.metadata_asset,role))
    with pytest.raises(ValueError,match='sparse-lookup'):EngramIQ4NLLookup(other,binding,role,tmp_path)
    identity=replace(binding.token_identity,compared_stored_map=False)
    with pytest.raises(ValueError,match='source-bound'):
        EngramIQ4NLLookup(bundle,replace(binding,token_identity=identity),asset,tmp_path)


@pytest.mark.parametrize('layer,hashes',[
    (True,torch.zeros(1,2,2,4,dtype=torch.int64)),
    (14,torch.zeros(1,2,2,4,dtype=torch.int64)),
    (1,torch.zeros(1,2,2,4,dtype=torch.int32)),
    (1,torch.zeros(1,2,2,3,dtype=torch.int64)),
    (1,torch.zeros(1,2,1,4,dtype=torch.int64)),
    (1,torch.zeros(1,2,4,dtype=torch.int64)),
])
def test_layer_and_hash_layout_must_be_explicit(tmp_path,layer,hashes):
    bundle,binding,asset,_=components(tmp_path)
    with EngramIQ4NLLookup(bundle,binding,asset,tmp_path) as lookup:
        with pytest.raises(ValueError):lookup.lookup_layer(layer,hashes)


def test_lookup_failure_cannot_commit_a_callers_hash_state(tmp_path):
    bundle,binding,asset,_=components(tmp_path)
    original=EngramHashState(binding.hash_spec,batch_size=1,max_sequence_tokens=8)
    preview=original.fork();hashes=preview(torch.tensor([[1,2,4]]),0)
    with EngramIQ4NLLookup(bundle,binding,asset,tmp_path,max_values=32) as lookup:
        with pytest.raises(ValueError,match='budget'):lookup.lookup_layer(1,hashes)
    assert original.position==0 and original._history==[[]] and preview.position==3


def test_bound_hash_rows_decoded_q5_bf16_and_f32_projection_component_chain(tmp_path):
    import struct
    from venturi.gguf_quantization import decode_q5_k_rows,decode_bf16_rows
    from venturi.models.engram_projection import reference_engram_from_rows
    from venturi.models.engram_residual import reference_engram_residual
    from test_gguf_projection_quantization import q5_block
    bundle,binding,asset,expected=components(tmp_path,head_dim=64)
    original=EngramHashState(binding.hash_spec,batch_size=1,max_sequence_tokens=8)
    preview=original.fork();hashes=preview(torch.tensor([[1,2,4]]),0)
    packed=[];golden=[]
    for row in range(40):
        raw,values=q5_block(d=(row%7+1)*2**-12,dmin=(row%5+1)*2**-13)
        packed.append(raw);golden.append(values)
    w=decode_q5_k_rows(b''.join(packed),256);golden_w=torch.stack(golden)
    q=decode_bf16_rows(struct.pack('<'+'H'*32,*([0x3f80]*32)),8)
    k=decode_bf16_rows(struct.pack('<'+'H'*32,*([0xbf00]*32)),8)
    x=torch.randn(1,3,4,8,generator=torch.Generator().manual_seed(417164))
    with EngramIQ4NLLookup(bundle,binding,asset,tmp_path) as lookup:
        rows=lookup.lookup_layer(1,hashes)
        actual=reference_engram_from_rows(x,rows,w,q,k,norm_epsilon=binding.config_norm_epsilon)
        reference_rows=expected['blk.1.engram_embd.weight'][hashes[:,:,0,:]]
        projected=reference_rows.reshape(3,256)@golden_w.T
        key=projected[:,:32].reshape(1,3,4,8);value=projected[:,32:].reshape(1,3,8)
        expected_output=reference_engram_residual(x,key,value,torch.ones(4,8),torch.full((4,8),-.5),
                                                  norm_epsilon=1e-20)
        assert torch.equal(actual,expected_output)
        w[0,0]=float('nan')
        with pytest.raises(ValueError):reference_engram_from_rows(x,rows,w,q,k,norm_epsilon=1e-20)
    assert original.position==0 and original._history==[[]]  # caller must explicitly commit preview only after success
