"""Tiny synthetic Q5_K/BF16 format fixtures, no real projection weights."""
import struct

import pytest
import torch

from venturi.gguf_quantization import decode_bf16_rows,decode_q5_k_rows


def q5_block(d=.5,dmin=.25,scales=None,mins=None,values=None):
    scales=list(range(8)) if scales is None else scales
    mins=list(reversed(range(8))) if mins is None else mins
    values=[[((group*4+column)%32) for column in range(32)] for group in range(8)] if values is None else values
    packed=bytearray(12)
    for i in range(4):
        packed[i]=(scales[i]&63)|((scales[i+4]>>4)<<6)
        packed[i+4]=(mins[i]&63)|((mins[i+4]>>4)<<6)
        packed[i+8]=(scales[i+4]&15)|((mins[i+4]&15)<<4)
    qh=bytes(sum(((values[group][column]>>4)&1)<<group for group in range(8)) for column in range(32))
    ql=bytes((values[pair*2][column]&15)|((values[pair*2+1][column]&15)<<4)
             for pair in range(4) for column in range(32))
    raw=struct.pack('<ee',d,dmin)+packed+qh+ql
    expected=torch.tensor([d*scales[g]*v-dmin*mins[g] for g in range(8) for v in values[g]],dtype=torch.float32)
    assert len(raw)==176
    return raw,expected


def test_q5_k_scale_min_fields_high_bits_and_column_order():
    scales=[63,62,32,31,16,15,1,0];mins=list(reversed(scales))
    raw,expected=q5_block(scales=scales,mins=mins)
    actual=decode_q5_k_rows(raw,256)
    assert torch.equal(actual[0],expected) and actual.dtype==torch.float32 and actual.device.type=='cpu'


def test_q5_high_bit_is_not_missing_or_confused_with_nibble_order():
    values=[[16 if group==column%8 else 0 for column in range(32)] for group in range(8)]
    raw,expected=q5_block(d=1.,dmin=0.,scales=[1]*8,values=values)
    assert torch.equal(decode_q5_k_rows(raw,256)[0],expected)
    assert expected.sum().item()==512.


def test_q5_multiple_rows_blocks_negative_zero_and_subnormal_scales():
    pairs=[q5_block(d=d,dmin=m) for d,m in ((1.,.5),(-2.,.25),(2**-24,-2**-24),(-0.,0.))]
    raw=b''.join(p[0] for p in pairs);expected=torch.cat([p[1] for p in pairs]).reshape(2,512)
    actual=decode_q5_k_rows(raw,512)
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    actual.fill_(777)
    assert torch.equal(decode_q5_k_rows(raw,512),expected)


@pytest.mark.parametrize('d,m',[(float('nan'),0.),(0.,float('nan')),(float('inf'),1.),(1.,-float('inf'))])
def test_q5_nonfinite_scale_and_min_are_not_silently_cleaned(d,m):
    raw,_=q5_block(d=d,dmin=m)
    with pytest.raises(ValueError,match='Nonfinite'):decode_q5_k_rows(raw,256)


def test_bf16_exact_bit_expansion_includes_signed_zero_subnormals_and_extremes():
    words=[0,0x8000,1,0x8001,0x3f80,0xbf80,0x7f7f,0xff7f]
    raw=struct.pack('<8H',*words);actual=decode_bf16_rows(raw,4)
    expected=torch.tensor([w<<16 if w<0x8000 else (w<<16)-2**32 for w in words],dtype=torch.int32)
    assert torch.equal(actual.view(torch.int32).flatten(),expected)
    assert actual.shape==(2,4) and actual.dtype==torch.float32 and actual.device.type=='cpu'


@pytest.mark.parametrize('word',[0x7f80,0xff80,0x7fc0,0x7f81,0xffc1])
def test_bf16_nonfinite_weight_patterns_rejected(word):
    with pytest.raises(ValueError,match='Nonfinite'):decode_bf16_rows(struct.pack('<H',word),1)


@pytest.mark.parametrize('decode,width,size',[(decode_q5_k_rows,256,176),(decode_bf16_rows,4,8)])
def test_explicit_formats_reject_empty_partial_mutable_rows_and_output_over_budget(decode,width,size):
    for raw in (b'',b'0'*(size-1),b'0'*(size+1),bytearray(size)):
        with pytest.raises(ValueError):decode(raw,width)
    for budget in (0,True,width-1,2**20+1):
        with pytest.raises(ValueError):decode(bytes(size),width,max_values=budget)
    for invalid in (True,0,-1,1.,2**16+256):
        with pytest.raises(ValueError):decode(bytes(size),invalid)


@pytest.mark.parametrize('width',[1,32,255,257])
def test_q5_rows_cannot_use_iq4nl_block_geometry(width):
    with pytest.raises(ValueError):decode_q5_k_rows(bytes(176),width)
