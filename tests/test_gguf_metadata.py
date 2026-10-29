import hashlib
import struct
import subprocess
import sys

import pytest

from venturi.gguf_metadata import read_verified_integer_metadata
from venturi.model_assets import ModelAsset


def write_gguf(root,tensors,metadata=()):
    payload=bytearray();entries=[]
    for name,typ,shape,values in tensors:
        offset=len(payload);fmt={0:'f',24:'b',25:'h',26:'i',27:'q'}[typ]
        data=struct.pack('<'+fmt*len(values),*values)
        payload.extend(data);payload.extend(bytes((-len(payload))%32))
        entries.append((name,typ,tuple(reversed(shape)),offset))
    header=b'GGUF'+struct.pack('<IQQ',3,len(entries),len(metadata))
    for name,typ,value in metadata:
        key=name.encode();header+=struct.pack('<Q',len(key))+key+struct.pack('<I',typ)
        if typ==8:
            raw=value.encode();header+=struct.pack('<Q',len(raw))+raw
        elif typ==9:
            et,items=value;fmt={0:'B',4:'I',5:'i',10:'Q',11:'q',6:'f'}[et]
            header+=struct.pack('<IQ',et,len(items))+struct.pack('<'+fmt*len(items),*items)
        else:header+=struct.pack('<'+{4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[typ],value)
    for name,typ,dims,offset in entries:
        label=name.encode();header+=struct.pack('<Q',len(label))+label+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',typ,offset)
    raw=header+bytes((-len(header))%32)+payload
    (root/'body.gguf').write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest();rev='a'*40
    return ModelAsset('body.gguf',len(raw),sha,f'https://huggingface.co/o/m/resolve/{rev}/body.gguf','o/m',rev,'body.gguf','backbone')


@pytest.fixture
def asset(tmp_path):
    return write_gguf(tmp_path,[('map',26,(5,),(0,1,2,1,0)),('wide',27,(2,2),(-1,2**62,3,4)),('matrix',0,(2,),(1.,2.))])


def test_explicit_full_sha_before_integer_decode_and_dtype_axis_order(tmp_path,asset):
    result=read_verified_integer_metadata(asset,tmp_path,['wide','map'])
    assert result.asset==asset and len(result.header_sha256)==64
    assert result.tensor('wide').shape==(2,2) and result.tensor('wide').values==(-1,2**62,3,4)
    assert result.tensor('map').values==(0,1,2,1,0)
    with pytest.raises(StopIteration):result.tensor('not there')


def test_same_length_wrong_payload_is_not_metadata_qualification(tmp_path,asset):
    p=tmp_path/'body.gguf';raw=p.read_bytes();p.write_bytes(raw[:-1]+b'1')
    with pytest.raises(ValueError,match='SHA256'):read_verified_integer_metadata(asset,tmp_path,['map'])


def test_partial_length_rejected_without_partial_fallback(tmp_path,asset):
    p=tmp_path/'body.gguf';p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError,match='exact manifest size'):read_verified_integer_metadata(asset,tmp_path,['map'])


@pytest.mark.parametrize('names',[[],['map','map'],['absent'],['matrix']])
def test_missing_duplicate_and_floating_metadata_fail_closed(tmp_path,asset,names):
    with pytest.raises(ValueError):read_verified_integer_metadata(asset,tmp_path,names)


@pytest.mark.parametrize('options',[{'max_payload_bytes':True},{'max_payload_bytes':-1},
    {'max_elements':2_000_001},{'max_elements':True},{'max_payload_bytes':19},{'max_elements':4}])
def test_payload_and_allocation_budgets_are_explicit(tmp_path,asset,options):
    with pytest.raises(ValueError):read_verified_integer_metadata(asset,tmp_path,['map'],**options)


def test_symlink_asset_refused(tmp_path,asset):
    p=tmp_path/'body.gguf';other=tmp_path/'other';p.rename(other);p.symlink_to(other)
    with pytest.raises(ValueError,match='Symlinked'):read_verified_integer_metadata(asset,tmp_path,['map'])


def test_changed_file_during_scan_is_retained_as_failure(tmp_path,asset,monkeypatch):
    import venturi.gguf_metadata as module
    real=module.hashlib.sha256;count=0
    class ChangingDigest:
        def __init__(self):self.inner=real()
        def update(self,data):
            self.inner.update(data)
            p=tmp_path/'body.gguf'
            with p.open('r+b') as f:f.seek(-1,2);f.write(b'9')
        def hexdigest(self):return self.inner.hexdigest()
    def hasher():
        nonlocal count
        count+=1
        return ChangingDigest() if count==2 else real()
    # Inventory allocates one hasher; the full scan allocates the second.
    monkeypatch.setattr(module.hashlib,'sha256',hasher)
    with pytest.raises(ValueError,match='changed'):read_verified_integer_metadata(asset,tmp_path,['map'])


def test_integer_metadata_import_is_torch_free():
    subprocess.check_call([sys.executable,'-c','import sys;import venturi.gguf_metadata;assert "torch" not in sys.modules'])



def test_inventory_header_must_match_same_verified_descriptor(tmp_path,asset,monkeypatch):
    import venturi.gguf_metadata as module
    from dataclasses import replace
    original=module.inspect_gguf
    def wrong_header(*args,**kwargs):
        return replace(original(*args,**kwargs),header_sha256='0'*64)
    monkeypatch.setattr(module,'inspect_gguf',wrong_header)
    with pytest.raises(ValueError,match='header differs'):
        read_verified_integer_metadata(asset,tmp_path,['map'])



def test_name_iterators_are_bounded_before_payload_work(tmp_path,asset):
    from itertools import repeat
    with pytest.raises(ValueError):read_verified_integer_metadata(asset,tmp_path,repeat('map'))
    with pytest.raises(ValueError):read_verified_integer_metadata(asset,tmp_path,'map')
    assert read_verified_integer_metadata(asset,tmp_path,iter(['map'])).tensor('map').values==(0,1,2,1,0)



def test_verified_header_arrays_keep_gguf_type_and_layer_order(tmp_path):
    a=write_gguf(tmp_path,[('map',26,(2,),(0,1))],
        [('layers',9,(4,(1,14))),('arch',8,'deepseek41'),('eps',6,1e-20)])
    result=read_verified_integer_metadata(a,tmp_path,['map'],header_array_names=['layers'])
    assert result.header_array('layers').element_type==4
    assert result.header_array('layers').values==(1,14)
    assert dict(result.scalars)['arch']=='deepseek41'
    assert dict(result.scalars)['eps']==struct.unpack('<f',struct.pack('<f',1e-20))[0]


@pytest.mark.parametrize('options',[
    {'header_array_names':['missing']},{'header_array_names':['arch']},
    {'header_array_names':['layers','layers']},{'header_array_names':['layers'],'max_header_array_elements':1},
    {'header_array_names':'layers'},{'max_header_array_elements':True}])
def test_header_array_schema_and_limits_fail_closed(tmp_path,options):
    a=write_gguf(tmp_path,[('map',26,(2,),(0,1))],[('layers',9,(4,(1,14))),('arch',8,'deepseek41')])
    with pytest.raises(ValueError):read_verified_integer_metadata(a,tmp_path,['map'],**options)


def test_floating_header_array_cannot_supply_integer_layer_ids(tmp_path):
    a=write_gguf(tmp_path,[('map',26,(2,),(0,1))],[('layers',9,(6,(1.,14.)))])
    with pytest.raises(ValueError,match='type/element'):
        read_verified_integer_metadata(a,tmp_path,['map'],header_array_names=['layers'])
