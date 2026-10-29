"""Tiny generated IQ4_NL tables only. No real model weights or target I/O."""
from dataclasses import replace
import hashlib
from itertools import repeat
import os
import struct
import subprocess
import sys

import pytest
import torch

from venturi.gguf_rows import VerifiedIQ4NLRows
from venturi.model_assets import ModelAsset


LUT=(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)
SCHEMA={'layer1':(11,64),'layer14':(7,32)}


def fixture(root, *, tables=None, metadata=()):
    if tables is None:tables=[('layer1',11,64,20),('layer14',7,32,20)]
    payload=bytearray();entries=[];expected={}
    for name,rows,width,typ in tables:
        offset=len(payload);values=[]
        for row in range(rows):
            line=[]
            for block in range(width//32):
                scale=float(row+1)*(1 if block%2==0 else -.5)
                payload.extend(struct.pack('<e',scale)+bytes((i+row)%16|(((15-i+block)%16)<<4) for i in range(16)))
                line.extend(scale*LUT[(i+row)%16] for i in range(16))
                line.extend(scale*LUT[(15-i+block)%16] for i in range(16))
            values.append(line)
        expected[name]=torch.tensor(values,dtype=torch.float32)
        payload.extend(bytes((-len(payload))%32));entries.append((name,rows,width,typ,offset))
    header=b'GGUF'+struct.pack('<IQQ',3,len(entries),len(metadata))
    for key,value in metadata:
        key=key.encode();value=value.encode()
        header+=struct.pack('<Q',len(key))+key+struct.pack('<IQ',8,len(value))+value
    for name,rows,width,typ,offset in entries:
        raw=name.encode();header+=struct.pack('<Q',len(raw))+raw+struct.pack('<IQQIQ',2,width,rows,typ,offset)
    raw=header+bytes((-len(header))%32)+payload
    (root/'lookup.gguf').write_bytes(raw);rev='a'*40
    asset=ModelAsset('lookup.gguf',len(raw),hashlib.sha256(raw).hexdigest(),
        f'https://huggingface.co/venturi-fixtures/rows/resolve/{rev}/lookup.gguf',
        'venturi-fixtures/rows',rev,'lookup.gguf','sparse_lookup')
    return asset,expected


def store(root,**kw):
    asset,expected=fixture(root)
    return VerifiedIQ4NLRows(asset,root,expected_shapes=SCHEMA,**kw),expected


def test_two_tables_one_source_and_merged_requests_preserve_shapes_and_order(tmp_path):
    reader,expected=store(tmp_path)
    with reader:
        ids=torch.tensor([[4,2],[4,3]])
        other=torch.tensor([3,2,4])
        a,b=reader.lookup_many('layer1',[ids,other])
        assert torch.equal(a,expected['layer1'][ids]) and torch.equal(b,expected['layer1'][other])
        assert a.shape==(2,2,64) and b.shape==(3,64)
        stat=reader.last_read
        assert (stat.requested_rows,stat.unique_rows,stat.payload_bytes,stat.read_calls)==(7,3,3*36,1)
        assert torch.equal(reader.lookup('layer14',torch.tensor([6,0])),expected['layer14'][[6,0]])
    assert reader.closed
    reader.close()
    with pytest.raises(ValueError,match='closed'):reader.lookup('layer1',ids)


def test_deduplication_noncontiguous_ids_output_aliases_and_no_persistent_row_cache(tmp_path):
    reader,expected=store(tmp_path)
    with reader:
        ids=torch.tensor([[0,2],[4,0]]).T
        a,b=reader.lookup_many('layer1',[ids,ids])
        stat=reader.last_read
        assert (stat.requested_rows,stat.unique_rows,stat.read_calls)==(8,3,3)
        assert torch.equal(a,expected['layer1'][ids])
        a.fill_(99)
        assert torch.equal(b,expected['layer1'][ids])
        assert torch.equal(reader.lookup('layer1',ids),b)
        assert reader.last_read.payload_bytes==3*36  # no hidden guaranteed resident cache


def test_input_ids_and_shape_are_snapshotted_before_payload_io(tmp_path,monkeypatch):
    import venturi.gguf_rows as module
    reader,expected=store(tmp_path);ids=torch.tensor([1,4]);original=module.os.pread
    def change_input(*args):
        ids.resize_(1,2);ids.fill_(0)
        return original(*args)
    monkeypatch.setattr(module.os,'pread',change_input)
    with reader:
        actual=reader.lookup('layer1',ids)
        assert actual.shape==(2,64) and torch.equal(actual,expected['layer1'][[1,4]])


def test_empty_requests_and_all_empty_group_have_no_payload_reads(tmp_path,monkeypatch):
    import venturi.gguf_rows as module
    reader,_=store(tmp_path)
    def forbidden(*args):raise AssertionError('Empty lookup must not pread')
    monkeypatch.setattr(module.os,'pread',forbidden)
    with reader:
        a,b=reader.lookup_many('layer1',[torch.empty((2,0),dtype=torch.int64),torch.empty(0,dtype=torch.int64)])
        assert a.shape==(2,0,64) and b.shape==(0,64) and reader.last_read.payload_bytes==0


@pytest.mark.parametrize('ids',[torch.tensor([-1]),torch.tensor([11]),torch.tensor([1.]),
    torch.tensor([True]),torch.tensor(1),torch.empty(1,dtype=torch.int64,device='meta'),[1]])
def test_invalid_ids_are_rejected_before_payload_io_and_do_not_poison_source(tmp_path,monkeypatch,ids):
    import venturi.gguf_rows as module
    reader,_=store(tmp_path);original=module.os.pread
    def forbidden(*args):raise AssertionError('Bad ID must not pread')
    monkeypatch.setattr(module.os,'pread',forbidden)
    with reader:
        with pytest.raises(ValueError):reader.lookup('layer1',ids)
        assert reader.last_read is None
        monkeypatch.setattr(module.os,'pread',original)
        assert reader.lookup('layer1',torch.tensor([0])).shape==(1,64)


def test_budgets_and_request_count_checked_before_io(tmp_path):
    reader,_=store(tmp_path,max_values=128)
    with reader:
        with pytest.raises(ValueError,match='budget'):reader.lookup('layer1',torch.tensor([0,0,0]))
        with pytest.raises(ValueError,match='request'):reader.lookup_many('layer1',repeat(torch.tensor([0])))
        with pytest.raises(ValueError,match='request'):reader.lookup_many('layer1',[])
        with pytest.raises(ValueError,match='request'):reader.lookup_many('layer1',torch.tensor([[0]]))
        with pytest.raises(ValueError,match='table name'):reader.lookup('not-admitted',torch.tensor([0]))


@pytest.mark.parametrize('schema',[{}, {'layer1':(12,64)}, {'layer1':(11,32)},
    {'layer1':(11,True)}, {'layer1':(11,33)}, {'layer1':(0,64)}, {'layer1':(11,64,1)},
    {'unknown':(11,64)}, {'layer1':[11.,64]}, {'layer1':(11,64*2048)}])
def test_explicit_dtype_shape_name_and_schema_are_required(tmp_path,schema):
    asset,_=fixture(tmp_path)
    with pytest.raises(ValueError):VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=schema)


def test_uniform_q4_with_same_byte_geometry_is_not_iq4nl(tmp_path):
    asset,_=fixture(tmp_path,tables=[('layer1',11,64,2)])
    with pytest.raises(ValueError,match='dtype/shape'):
        VerifiedIQ4NLRows(asset,tmp_path,expected_shapes={'layer1':(11,64)})


def test_missing_full_sha_partial_symlink_and_nonregular_sources_are_rejected(tmp_path):
    asset,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='SHA256'):
        VerifiedIQ4NLRows(replace(asset,sha256='0'*64),tmp_path,expected_shapes=SCHEMA)
    p=tmp_path/asset.filename;raw=p.read_bytes();p.write_bytes(raw[:-1])
    with pytest.raises(ValueError,match='exact manifest size'):VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=SCHEMA)
    p.unlink();p.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError,match='Symlinked'):VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=SCHEMA)
    p.unlink();os.mkfifo(p)
    with pytest.raises(ValueError,match='regular source'):VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=SCHEMA)


def test_same_length_payload_corruption_after_admission_poisoned_not_retried(tmp_path):
    reader,_=store(tmp_path)
    with reader:
        with (tmp_path/'lookup.gguf').open('r+b') as f:f.seek(-1,2);f.write(b'1')
        with pytest.raises(ValueError,match='changed'):reader.lookup('layer1',torch.tensor([0]))
        with pytest.raises(ValueError,match='lease failed'):reader.lookup('layer1',torch.tensor([1]))


@pytest.mark.parametrize('mode',['change','short','eio'])
def test_change_during_pread_short_reads_and_io_errors_do_not_return_plausible_rows(tmp_path,monkeypatch,mode):
    import venturi.gguf_rows as module
    reader,_=store(tmp_path);real=module.os.pread
    def failing(fd,size,offset):
        if mode=='eio':raise OSError(5,'injected EIO')
        if mode=='short':return b''
        raw=real(fd,size,offset)
        with (tmp_path/'lookup.gguf').open('r+b') as f:f.seek(-1,2);f.write(b'1')
        return raw
    monkeypatch.setattr(module.os,'pread',failing)
    with reader:
        with pytest.raises((ValueError,OSError)):reader.lookup('layer1',torch.tensor([0]))
        assert reader.last_read is None
        with pytest.raises(ValueError,match='lease failed'):reader.lookup('layer1',torch.tensor([0]))


def test_partial_pread_progress_is_completed_and_counted(tmp_path,monkeypatch):
    import venturi.gguf_rows as module
    reader,expected=store(tmp_path);real=module.os.pread
    monkeypatch.setattr(module.os,'pread',lambda fd,size,offset:real(fd,min(size,7),offset))
    with reader:
        assert torch.equal(reader.lookup('layer1',torch.tensor([0])),expected['layer1'][[0]])
        assert reader.last_read.read_calls==6 and reader.last_read.payload_bytes==36


def test_context_closes_descriptor_on_lookup_failure(tmp_path):
    reader,_=store(tmp_path)
    with pytest.raises(ValueError):
        with reader:reader.lookup('layer1',torch.tensor([99]))
    assert reader.closed


def test_header_must_match_the_full_sha_open_descriptor(tmp_path,monkeypatch):
    import venturi.gguf_rows as module
    asset,_=fixture(tmp_path);original=module.inspect_gguf
    def different(*args,**kw):return replace(original(*args,**kw),header_sha256='0'*64)
    monkeypatch.setattr(module,'inspect_gguf',different)
    with pytest.raises(ValueError,match='header differs'):
        VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=SCHEMA)


def test_full_sha_alone_does_not_satisfy_required_source_metadata(tmp_path):
    asset,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='scalar identity'):
        VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=SCHEMA,required_metadata={'general.architecture':'wrong'})


def test_concurrent_callers_are_serialized_but_outputs_remain_request_private(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    reader,expected=store(tmp_path)
    with reader,ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda row:reader.lookup('layer1',torch.tensor([row,row])),range(8)))
    for row,result in enumerate(results):assert torch.equal(result,expected['layer1'][[row,row]])
    results[0].fill_(999)
    assert torch.equal(results[1],expected['layer1'][[1,1]])


def test_admission_failure_closes_the_new_source_descriptor(tmp_path,monkeypatch):
    import venturi.gguf_rows as module
    asset,_=fixture(tmp_path);opened=[];real=module.os.fdopen
    def record(*args,**kw):
        stream=real(*args,**kw);opened.append(stream);return stream
    monkeypatch.setattr(module.os,'fdopen',record)
    with pytest.raises(ValueError,match='SHA256'):
        VerifiedIQ4NLRows(replace(asset,sha256='0'*64),tmp_path,expected_shapes=SCHEMA)
    assert opened and all(stream.closed for stream in opened)


def test_change_during_full_admission_scan_is_rejected(tmp_path,monkeypatch):
    import venturi.gguf_rows as module
    asset,_=fixture(tmp_path);original=module.hashlib.sha256;count=0
    class ChangingDigest:
        def __init__(self):self.inner=original()
        def update(self,data):
            self.inner.update(data)
            with (tmp_path/'lookup.gguf').open('r+b') as f:f.seek(-1,2);f.write(b'1')
        def hexdigest(self):return self.inner.hexdigest()
    def hasher():
        nonlocal count
        count+=1
        return ChangingDigest() if count==2 else original()
    monkeypatch.setattr(module.hashlib,'sha256',hasher)
    with pytest.raises(ValueError,match='changed'):
        VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=SCHEMA)


def test_full_sha_valid_but_nonfinite_quantized_scale_poisoned_not_cleaned(tmp_path):
    from venturi.gguf_inventory import inspect_gguf
    asset,_=fixture(tmp_path);path=tmp_path/asset.filename
    offset=inspect_gguf(path).tensors[0].offset
    with path.open('r+b') as f:f.seek(offset);f.write(struct.pack('<e',float('inf')))
    # This new GENERATED manifest authenticates the bad bytes; it does not make
    # their numerical meaning valid. Do not rewrite/clean that scale at lookup.
    asset=replace(asset,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with VerifiedIQ4NLRows(asset,tmp_path,expected_shapes=SCHEMA) as reader:
        with pytest.raises(ValueError,match='Nonfinite'):reader.lookup('layer1',torch.tensor([0]))
        assert reader.last_read is None
        with pytest.raises(ValueError,match='lease failed'):reader.lookup('layer1',torch.tensor([1]))


def test_import_has_no_torch_or_gpu_initialization():
    subprocess.check_call([sys.executable,'-c','import sys;import venturi.gguf_rows;assert "torch" not in sys.modules'])
