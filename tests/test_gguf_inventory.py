import hashlib
import os
import struct
import subprocess
import sys

import pytest

from venturi.gguf_inventory import GgufArraySummary, GgufLimits, inspect_gguf


def string(value):
    raw=value.encode() if isinstance(value,str) else value
    return struct.pack('<Q',len(raw))+raw


def fixture(*, metadata=(), tensors=None, alignment=32, version=3, payload=True):
    if tensors is None:
        tensors=[('dense',1,(32,2),0),('experts',29,(256,2,3),128)]
    header=b'GGUF'+struct.pack('<IQQ',version,len(tensors),len(metadata))
    for name,typ,value in metadata:
        header+=string(name)+struct.pack('<I',typ)+value
    for name,typ,dims,offset in tensors:
        header+=string(name)+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',typ,offset)
    total=((len(header)+alignment-1)//alignment*alignment if tensors else len(header))+480
    return header+b'\0'*(total-len(header)) if payload else header, len(header),total


def write(tmp_path, **changes):
    data,header,size=fixture(**changes)
    p=tmp_path/'model.gguf';p.write_bytes(data)
    return p,header,size


def test_mixed_layout_spans_axes_and_hash_without_tensor_read(tmp_path):
    p,header,size=write(tmp_path,metadata=[('general.architecture',8,string('deepseek41'))])
    v=inspect_gguf(p,expected_bytes=size)
    assert v.metadata['general.architecture']=='deepseek41'
    assert v.header_bytes_read==header and v.header_sha256==hashlib.sha256(p.read_bytes()[:header]).hexdigest()
    assert [t.dtype for t in v.tensors]==['F16','IQ1_M']
    assert v.tensors[1].shape==(3,2,256) and v.tensors[1].dimensions_ggml==(256,2,3)
    assert v.tensor_bytes==128+336 and v.tensors[1].offset==v.data_offset+128
    assert v.file_length_complete and not v.payload_sha256_verified


def test_partial_header_requires_independent_bound_and_never_verified(tmp_path):
    p,header,size=write(tmp_path,payload=False)
    with pytest.raises(ValueError,match='expected file bytes'): inspect_gguf(p,allow_partial=True)
    with pytest.raises(ValueError,match='length'): inspect_gguf(p,expected_bytes=size)
    v=inspect_gguf(p,allow_partial=True,expected_bytes=size)
    assert not v.file_length_complete and not v.payload_sha256_verified
    assert v.header_bytes_read==header and v.observed_file_bytes==header
    with pytest.raises(ValueError,match='exceed expected'): inspect_gguf(p,allow_partial=True,expected_bytes=header)


def test_metadata_arrays_are_summarized_not_materialized(tmp_path):
    values=struct.pack('<IQ3i',5,3,1,4,9)
    strings=struct.pack('<IQ',8,2)+string('hello')+string('世界')
    p,_,_=write(tmp_path,metadata=[('ints',9,values),('tokens',9,strings)])
    m=inspect_gguf(p).metadata
    assert m=={'ints':GgufArraySummary(5,3),'tokens':GgufArraySummary(8,2)}
    with pytest.raises(ValueError,match='aggregate array'): inspect_gguf(p,limits=GgufLimits(array_items=4))


@pytest.mark.parametrize('change,match',[
    ({'version':2},'version 3'),
    ({'metadata':[('x',4,struct.pack('<I',1)),('x',4,struct.pack('<I',2))]},'duplicate.*metadata'),
    ({'tensors':[('x',0,(2,),0),('x',0,(2,),32)]},'duplicate.*tensor'),
    ({'tensors':[('x',29,(255,2),0)]},'row width'),
    ({'tensors':[('x',999,(32,),0)]},'layout 999'),
    ({'tensors':[('x',0,(),0)]},'rank'),
    ({'tensors':[('x',0,(1,2,3,4,5),0)]},'rank'),
    ({'tensors':[('x',0,(0,),0)]},'dimensions'),
    ({'tensors':[('x',0,(2**62,4),0)]},'dimensions'),
    ({'tensors':[('x',0,(32,),32)]},'offset'),
    ({'tensors':[('x',0,(32,),0),('y',0,(32,),64)]},'offset'),
    ({'metadata':[('general.alignment',4,struct.pack('<I',3))]},'alignment'),
    ({'metadata':[('general.alignment',10,struct.pack('<Q',32))]},'alignment'),
    ({'metadata':[('general.alignment',4,struct.pack('<I',2**21))]},'alignment'),
    ({'metadata':[('x',13,b'')]},'metadata type'),
    ({'metadata':[('x',9,struct.pack('<IQ',9,1))]},'nested'),
    ({'metadata':[('x',7,b'\2')]},'bool'),
    ({'metadata':[('x',6,struct.pack('<f',float('nan')))]},'Nonfinite'),
    ({'metadata':[('x',8,string(b'\xff'))]},'UTF-8'),
])
def test_malformed_or_unsupported_headers_fail_closed(tmp_path,change,match):
    p,_,_=write(tmp_path,**change)
    with pytest.raises(ValueError,match=match): inspect_gguf(p)


@pytest.mark.parametrize('limit,match',[
    ({'header_bytes':24},'byte budget'),({'string_bytes':1},'string byte'),
    ({'tensor_entries':1},'entry count')])
def test_budgets_reject_before_allocation(tmp_path,limit,match):
    p,_,_=write(tmp_path)
    with pytest.raises(ValueError,match=match): inspect_gguf(p,limits=GgufLimits(**limit))


@pytest.mark.parametrize('cut',[0,4,10,23,24,30,80])
def test_truncated_prefix(tmp_path,cut):
    data,_,size=fixture();p=tmp_path/'partial';p.write_bytes(data[:cut])
    with pytest.raises(ValueError,match='Truncated'): inspect_gguf(p,allow_partial=True,expected_bytes=size)


@pytest.mark.parametrize('changes',[{'expected_bytes':True},{'expected_bytes':0},{'expected_bytes':2**64},
                                     {'allow_partial':1},{'limits':{} }])
def test_explicit_options_validated(tmp_path,changes):
    p,_,_=write(tmp_path)
    with pytest.raises(ValueError): inspect_gguf(p,**changes)


def test_symlink_and_nonregular_refused(tmp_path):
    p,_,_=write(tmp_path);link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError,match='Symlink'): inspect_gguf(link)
    fifo=tmp_path/'fifo';os.mkfifo(fifo)
    with pytest.raises(ValueError,match='regular'): inspect_gguf(fifo)


def test_only_header_is_read_not_payload(tmp_path,monkeypatch):
    import venturi.gguf_inventory as module
    p,header,size=write(tmp_path)
    original=module._HeaderReader.read;positions=[]
    def bounded(self,n):
        positions.append(self.position+n)
        assert self.position+n<=header
        return original(self,n)
    monkeypatch.setattr(module._HeaderReader,'read',bounded)
    assert inspect_gguf(p).header_bytes_read==max(positions)==header


def test_zero_tensor_metadata_file_does_not_round_header(tmp_path):
    p,header,_=write(tmp_path,tensors=[])
    v=inspect_gguf(p)
    assert v.data_offset==header and not v.tensors


def test_torch_free_import_in_fresh_process():
    out=subprocess.check_output([sys.executable,'-c','import sys;import venturi.gguf_inventory;assert "torch" not in sys.modules;assert "safetensors" not in sys.modules'],text=True)
    assert out==''


@pytest.mark.parametrize('typ,width,stored',[(0,1,4),(1,1,2),(8,32,34),(19,256,50),
    (20,32,18),(29,256,56),(30,1,2),(39,32,17),(43,32,33),(67,32,33),
    (26,1,4),(27,1,8)])
def test_block_bytes_include_scales_and_match_row_geometry(tmp_path,typ,width,stored):
    p,_,_=write(tmp_path,tensors=[('t',typ,(width*2,3,4),0)])
    # Supply independent manifest size because large synthetic shapes exceed
    # this fixture's deliberately small payload, not because they are verified.
    v=inspect_gguf(p,allow_partial=True,expected_bytes=100_000)
    assert v.tensors[0].bytes==2*3*4*stored


def test_unknown_big_count_is_rejected_from_fixed_header(tmp_path):
    p=tmp_path/'bad';p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,2**63,0))
    with pytest.raises(ValueError,match='entry count'): inspect_gguf(p)


def test_metadata_budget_and_bounded_numeric_array_skip(tmp_path):
    p,_,_=write(tmp_path,metadata=[('a',4,struct.pack('<I',1)),('b',4,struct.pack('<I',1))])
    with pytest.raises(ValueError,match='entry count'): inspect_gguf(p,limits=GgufLimits(metadata_entries=1))
    p,_,_=write(tmp_path,metadata=[('a',9,struct.pack('<IQ',10,2**50))])
    with pytest.raises(ValueError,match='array item'): inspect_gguf(p)
    p,_,_=write(tmp_path,metadata=[('a',9,struct.pack('<IQ',10,100))])
    with pytest.raises(ValueError,match='Truncated'): inspect_gguf(p)


@pytest.mark.parametrize('change',[{'header_bytes':True},{'array_items':0},{'alignment':-1}])
def test_limit_options_reject_nonpositive_or_boolean(change):
    with pytest.raises(ValueError):GgufLimits(**change)



def test_metadata_only_file_ends_at_unaligned_header_without_tensor_blob(tmp_path):
    p,header,_=write(tmp_path,tensors=[],payload=False,
        metadata=[('general.architecture',8,string('deepseek41'))])
    assert header%32!=0 and p.stat().st_size==header
    v=inspect_gguf(p,expected_bytes=header)
    assert v.data_offset==header and v.file_length_complete and not v.tensors
