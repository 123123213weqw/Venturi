"""Q4 codec/archive contracts, not quantized-model quality or GPU performance."""
import json

import pytest
import torch

from venturi.models.flash_next_q4 import CODEC, LazyQ4ExpertArchive, decode_q4_fp16, quantize_q4_rtn
from venturi.models.flash_next_moe import MoEGeometry, TensorExpertStore
from venturi.ops.flash_next_q4 import NativeQ4
from venturi.ops.weights import Q4RowSplitWeight


@pytest.mark.parametrize('shape',[(1,1),(3,63),(4,64),(2,65),(3,128),(2,131),(5,640)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_codec_decode_independent_nibbles_tail_and_owned(shape,dtype):
    torch.manual_seed(223)
    x = torch.randn(shape).to(dtype)
    before = x.clone()
    q = quantize_q4_rtn(x)
    independent = torch.empty(shape,dtype=torch.float16)
    for row in range(shape[0]):
        for col in range(shape[1]):
            byte = int(q.codes[row,col//2])
            code = (byte >> (4*(col%2))) & 15
            signed = code if code < 8 else code-16
            independent[row,col] = signed*float(q.scales[row,col//64])
    assert torch.equal(decode_q4_fp16(q),independent)
    assert torch.equal(x,before)
    assert q.nbytes == shape[0]*q.padded_k*(.5+2/64)
    assert not ((q.codes & 15)==8).any() and not ((q.codes >> 4)==8).any()
    x.zero_()
    assert torch.equal(decode_q4_fp16(q),independent)


def test_zero_tiny_ties_stored_scale_and_saturation():
    x = torch.zeros(2,128)
    x[0,:6] = torch.tensor([7.,.5,1.5,2.5,-.5,-1.5])
    x[1,0] = 1e-10
    q = quantize_q4_rtn(x)
    y = decode_q4_fp16(q)
    assert y[0,:6].tolist() == [7.,0.,2.,2.,0.,-2.]
    assert not y[1].any() and torch.isfinite(q.scales).all()
    assert q.scales[0,0] == 1
    z = torch.randn(2,128)*.317
    q = quantize_q4_rtn(z)
    expected = (z.reshape(2,2,64)/q.scales.float().unsqueeze(-1)).round().clamp(-7,7)
    unsigned = torch.stack((q.codes&15,q.codes>>4),-1).reshape(2,2,64)
    assert torch.equal((expected.to(torch.int16)&15),unsigned.to(torch.int16))


@pytest.mark.parametrize('bad',[torch.tensor([[float('nan')]]),torch.tensor([[float('inf')]]),
                              torch.tensor([[1e20]]),torch.empty(0,2),torch.ones(3),torch.ones(1,1,dtype=torch.int64)])
def test_quantizer_rejects_invalid_source(bad):
    with pytest.raises(ValueError):
        quantize_q4_rtn(bad)


def test_memory_and_library_admission(tmp_path):
    for size in [0,-1,True,1]:
        with pytest.raises(ValueError):
            quantize_q4_rtn(torch.ones(2,2),max_elements=size)
    for size in [0,-1,True]:
        with pytest.raises(ValueError):
            NativeQ4(tmp_path/'missing.so',max_output_bytes=size)
    with pytest.raises(FileNotFoundError):
        NativeQ4(tmp_path/'missing.so')
    native = NativeQ4.__new__(NativeQ4)
    native.max_output_bytes=100000
    for scales in [torch.tensor([[float('nan'),0]],dtype=torch.float16),torch.tensor([[-1.,0]],dtype=torch.float16)]:
        q = Q4RowSplitWeight(torch.zeros(1,64,dtype=torch.uint8),scales,8)
        with pytest.raises(ValueError):
            native.prepare(q,device='cuda:0')
    with pytest.raises(ValueError):
        native.linear(torch.ones(1,2),None)


def source():
    g = MoEGeometry(8,4,2,6,6)
    return TensorExpertStore(g,{e:(torch.randn(12,8).half(),torch.randn(8,6).half()) for e in range(4)})


def test_lazy_archive_atomic_reuse_hashes_budgets_and_missing(tmp_path):
    s = source()
    path = tmp_path/'layer'
    a = LazyQ4ExpertArchive(s,path,source_identity={'revision':'fixture'},create=True)
    first = a.packed(3)
    assert a.stats['converted']==1 and a.stats['retained_decoded_bytes']==0
    assert a.stats['encoded_bytes']==sum(w.nbytes for w in first)
    assert all(torch.equal(x,y) for x,y in zip(a.read(3,device='cpu',dtype=torch.float16),map(decode_q4_fp16,first)))
    assert a.stats['converted']==1
    b = LazyQ4ExpertArchive(s,path,source_identity={'revision':'fixture'})
    assert all(torch.equal(x.codes,y.codes) for x,y in zip(first,b.packed(3)))
    with pytest.raises(ValueError,match='mismatch'):
        LazyQ4ExpertArchive(s,path,source_identity={'revision':'other'})
    with pytest.raises(FileExistsError):
        LazyQ4ExpertArchive(s,path,source_identity={'revision':'fixture'},create=True)
    for e in [-1,4,True]:
        with pytest.raises(ValueError):
            a.packed(e)
    with pytest.raises(ValueError):
        a.read(3,device='cpu',dtype=torch.float32)
    payload=path/'expert-0003.safetensors'
    raw=payload.read_bytes();payload.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    c = LazyQ4ExpertArchive(s,path,source_identity={'revision':'fixture'})
    with pytest.raises(ValueError,match='checksum'):
        c.packed(3)


def test_archive_does_not_overwrite_uncommitted_partial(tmp_path):
    s = source()
    a = LazyQ4ExpertArchive(s,tmp_path/'l',source_identity={'revision':'fixture'},create=True)
    partial = a.directory/'expert-0000.safetensors.part'
    partial.write_bytes(b'evidence')
    with pytest.raises(ValueError,match='Interrupted'):
        a.packed(0)
    assert partial.read_bytes()==b'evidence'


def test_general_decoder_also_handles_minus8():
    q = Q4RowSplitWeight.from_codes(torch.tensor([[-8,7,-1,0]]),torch.tensor([[.5]],dtype=torch.float16))
    assert decode_q4_fp16(q).tolist()==[[-4.,3.5,-.5,0.]]
