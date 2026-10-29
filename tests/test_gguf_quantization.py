import struct
import subprocess
import sys

import pytest
import torch

from venturi.gguf_quantization import decode_iq4_nl_rows


LUT = (-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)


def block(scale, pairs=None):
    return struct.pack('<e', scale)+bytes(i|((15-i)<<4) for i in range(16)) if pairs is None else struct.pack('<e',scale)+pairs


def test_nonuniform_codebook_and_split_nibbles_not_interleaved_or_signed_int4():
    actual = decode_iq4_nl_rows(block(.5),32)
    expected = torch.tensor([.5*v for v in LUT+tuple(reversed(LUT))])
    assert torch.equal(actual[0], expected) and actual.dtype == torch.float32 and actual.device.type == 'cpu'
    assert actual[0,0].item() == -63.5 and actual[0,8].item() == .5


def test_multiple_rows_and_per_block_scales_not_shared_row_scale():
    scales=(.5, -2., 2**-24, 65504.)
    raw=b''.join(block(s) for s in scales)
    rows=decode_iq4_nl_rows(raw,64)
    assert rows.shape == (2,64)
    for i,scale in enumerate(scales):
        assert torch.equal(rows.flatten()[i*32:(i+1)*32],torch.tensor([scale*v for v in LUT+tuple(reversed(LUT))]))
    rows.fill_(99)
    assert decode_iq4_nl_rows(raw,64)[0,0].item()==-63.5


def test_signed_zero_and_float16_little_endian_are_preserved():
    positive=decode_iq4_nl_rows(block(0.),32);negative=decode_iq4_nl_rows(block(-0.),32)
    assert torch.equal(positive,negative) and torch.equal(torch.signbit(positive),~torch.signbit(negative))
    one=b'\x00\x3c'+bytes([0x88])*16
    assert torch.equal(decode_iq4_nl_rows(one,32),torch.ones(1,32))


@pytest.mark.parametrize('scale',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_scale_is_rejected_not_replaced(scale):
    with pytest.raises(ValueError,match='Nonfinite'):decode_iq4_nl_rows(block(scale),32)


@pytest.mark.parametrize('width',[0,-32,16,33,True,32.,2**16+32])
def test_invalid_or_excessive_width(width):
    with pytest.raises(ValueError):decode_iq4_nl_rows(block(1.),width)


@pytest.mark.parametrize('raw',[b'',b'0'*17,b'0'*19,bytearray(b'0'*18),memoryview(b'0'*18),'0'*18])
def test_partial_empty_or_mutable_rows_are_rejected(raw):
    with pytest.raises(ValueError):decode_iq4_nl_rows(raw,32)


@pytest.mark.parametrize('budget',[0,True,31,2**20+1])
def test_output_budgets_are_checked_before_allocating(budget):
    with pytest.raises(ValueError):decode_iq4_nl_rows(block(1.),32,max_values=budget)


def test_import_has_no_torch_or_gpu_side_effects():
    subprocess.check_call([sys.executable,'-c','import sys;import venturi.gguf_quantization;assert "torch" not in sys.modules'])
