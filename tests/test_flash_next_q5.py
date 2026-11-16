"""Q5 codec/archive contracts, not quantized-model quality or GPU performance."""
import os

import pytest
import torch

from venturi.models.flash_next.flash_next_q5 import CODEC, decode_q5_fp16, quantize_q5_rtn
from venturi.ops.flash_next_q5 import NativeQ5
from venturi.ops.weights import Q5RowSplitWeight


@pytest.mark.parametrize('shape',[(1,1),(3,63),(4,64),(2,65),(3,128),(2,131),(5,640)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_codec_decode_independent_codes_tail_and_owned(shape,dtype):
    torch.manual_seed(229)
    x = torch.randn(shape).to(dtype)
    before = x.clone()
    q = quantize_q5_rtn(x)
    independent = torch.empty(shape,dtype=torch.float16)
    for row in range(shape[0]):
        for col in range(shape[1]):
            byte = int(q.codes[row,col//2])
            code = (byte >> (4*(col%2))) & 15
            group, lane = col//64, col%64
            hb = (int(q.high[row,group*8+lane//8]) >> (lane%8)) & 1
            signed = (code | (hb << 4))
            signed = signed if signed < 16 else signed-32
            independent[row,col] = signed*float(q.scales[row,group])
    assert torch.equal(decode_q5_fp16(q),independent)
    assert torch.equal(x,before)
    assert q.nbytes == shape[0]*q.padded_k*(.5+1/8+2/64)
    x.zero_()
    assert torch.equal(decode_q5_fp16(q),independent)


def test_full_code_range_endpoints_and_ties():
    torch.manual_seed(7)
    x = torch.randn(4,128)*.317
    q = quantize_q5_rtn(x)
    unsigned = torch.stack((q.codes&15,q.codes>>4),-1).reshape(4,2,64)
    bytes_ = q.high.view(4,2,8)
    bits = (bytes_.unsqueeze(-1).to(torch.int16) >> torch.arange(8,dtype=torch.int16)) & 1
    lanes = (unsigned.to(torch.int16) | (bits.reshape(4,2,64) << 4))
    assert lanes.min() >= 0 and lanes.max() <= 31
    # RTN with scale max|w|/15 saturates at +15/-16; +16 is unreachable.
    expected = (x.reshape(4,2,64)/q.scales.float().unsqueeze(-1)).round().clamp(-16,15)
    assert torch.equal((expected.to(torch.int16)&31),lanes.to(torch.int16))
    q2 = Q5RowSplitWeight.from_codes(
        torch.tensor([[-16,-1,0,1,15,15,-8,8]]),torch.tensor([[.5]],dtype=torch.float16))
    assert decode_q5_fp16(q2).tolist() == [[-8.,-.5,0.,.5,7.5,7.5,-4.,4.]]


def test_from_codes_roundtrip_all_thirty_two_codes():
    codes = torch.arange(-16,16,dtype=torch.int16).reshape(1,32)
    q = Q5RowSplitWeight.from_codes(codes,torch.tensor([[.25]],dtype=torch.float16))
    assert decode_q5_fp16(q).tolist() == [[c*.25 for c in range(-16,16)]]
    assert q.dequantize_reference().flatten().tolist() == [c*.25 for c in range(-16,16)]


def test_zero_group_and_saturation_scale():
    x = torch.zeros(2,128)
    # Group max 7.5 makes the stored scale exactly 0.5 (fp16); codes decode to
    # 15, 0 (tie), 2, 8 (tie), 0, -15 -> values 7.5, 0, 1, 4, 0, -7.5.
    x[0,:6] = torch.tensor([7.5,.25,.75,3.75,-.25,-7.5])
    x[1,0] = 1e-10
    q = quantize_q5_rtn(x)
    y = decode_q5_fp16(q)
    assert y[0,:6].tolist() == [7.5,0.,1.,4.,0.,-7.5]
    assert not y[1].any() and torch.isfinite(q.scales).all()
    assert q.scales[0,0] == 0.5


@pytest.mark.parametrize('bad',[torch.tensor([[float('nan')]]),torch.tensor([[float('inf')]]),
                                torch.tensor([[1e20]]),torch.empty(0,2),torch.ones(3),
                                torch.ones(1,1,dtype=torch.int64)])
def test_quantizer_rejects_invalid_source(bad):
    with pytest.raises(ValueError):
        quantize_q5_rtn(bad)


def test_descriptor_validation_and_admission(tmp_path):
    with pytest.raises(ValueError):
        Q5RowSplitWeight(torch.zeros(1,64,dtype=torch.uint8),torch.zeros(1,16,dtype=torch.uint8),
                         torch.zeros(1,2,dtype=torch.float32),64)
    with pytest.raises(ValueError):
        Q5RowSplitWeight.from_codes(torch.tensor([[16]]),torch.tensor([[.5]],dtype=torch.float16))
    with pytest.raises(ValueError):
        Q5RowSplitWeight.from_codes(torch.tensor([[-17]]),torch.tensor([[.5]],dtype=torch.float16))
    for size in [0,-1,True,1]:
        with pytest.raises(ValueError):
            quantize_q5_rtn(torch.ones(2,2),max_elements=size)
    for size in [0,-1,True]:
        with pytest.raises(ValueError):
            NativeQ5(tmp_path/'missing.so',max_output_bytes=size)
    with pytest.raises(FileNotFoundError):
        NativeQ5(tmp_path/'missing.so')
    native = NativeQ5.__new__(NativeQ5)
    native.max_output_bytes=100000
    good_codes = torch.zeros(1,64,dtype=torch.uint8)
    good_high = torch.zeros(1,16,dtype=torch.uint8)
    for scales in [torch.tensor([[float('nan'),0]],dtype=torch.float16),
                   torch.tensor([[-1.,0]],dtype=torch.float16)]:
        q = Q5RowSplitWeight(good_codes,good_high,scales,128)
        with pytest.raises(ValueError):
            native.prepare(q,device='cuda:0')
    with pytest.raises(ValueError):
        native.linear(torch.ones(1,2,dtype=torch.float16),None)


@pytest.mark.skipif(not torch.cuda.is_available() or
                    torch.cuda.get_device_capability(0) != (7,0) or
                    not os.environ.get('VENTURI_Q5_LIBRARY'),
                    reason='explicit SM70 device and VENTURI_Q5_LIBRARY required')
class TestNativeQ5Execution:
    """Native execution checks. Host tensors are built/computed on CPU and
    copied to the device: this host's torch build may lack SM70 compute
    kernels, while the operator library itself targets sm_70 only."""

    @classmethod
    def setup_class(cls):
        cls.native = NativeQ5(os.environ['VENTURI_Q5_LIBRARY'])

    def test_decode_and_linear_match_reference(self):
        torch.manual_seed(31)
        w = torch.randn(64,320)
        q = quantize_q5_rtn(w)
        prepared = self.native.prepare(q,device='cuda:0')
        reference = decode_q5_fp16(q).to('cuda:0')
        assert torch.equal(self.native.decode(prepared).cpu(),decode_q5_fp16(q))
        for tokens in (1,3,17,64):
            x = torch.randn(tokens,320,dtype=torch.float16).to('cuda:0')
            reference_cpu = decode_q5_fp16(q)
            expected = (x.cpu().float()@reference_cpu.float().t()).half()
            got = self.native.linear(x,prepared).cpu()
            # Warp butterfly FP32 reduction ordering differs from the torch dot;
            # compare against a small relative bound, not exact equality.
            peak = expected.float().abs().max().clamp_min(1e-6)
            assert float((got.float()-expected.float()).abs().max()/peak) < 1e-3

    def test_bounded_tokens_rejected(self):
        torch.manual_seed(32)
        q = quantize_q5_rtn(torch.randn(8,128))
        prepared = self.native.prepare(q,device='cuda:0')
        with pytest.raises(ValueError):
            self.native.linear(torch.randn(65,128,dtype=torch.float16).to('cuda:0'),prepared)
