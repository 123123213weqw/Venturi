import struct
import subprocess
import sys

import pytest
import torch

from venturi.container.gguf.gguf_quantization import decode_iq4_nl_rows


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
    subprocess.check_call([sys.executable,'-c','import sys;import venturi.container.gguf.gguf_quantization;assert "torch" not in sys.modules'])


def test_q8_0_q6_k_iq2_s_iq3_s_geometry_and_rejection():
    from venturi.container.gguf.gguf_quantization import (
        decode_q8_0_rows, decode_q6_k_rows, decode_iq2_s_rows,
        decode_iq3_s_rows,
    )
    cases = (
        (decode_q8_0_rows, 32, 34), (decode_q6_k_rows, 256, 210),
        (decode_iq2_s_rows, 256, 82), (decode_iq3_s_rows, 256, 110),
    )
    for decoder, block_values, block_bytes in cases:
        with pytest.raises(ValueError):
            decoder(bytes(block_bytes - 1), block_values)  # short block
        with pytest.raises(ValueError):
            decoder(bytes(block_bytes * 2), block_values + 1)  # width not aligned


def test_q8_0_hand_constructed_block():
    import struct as struct_module
    from venturi.container.gguf.gguf_quantization import decode_q8_0_rows
    block = struct_module.pack("<e", 0.5) + bytes(range(32))
    expected = [0.5 * (value if value < 128 else value - 256) for value in range(32)]
    out = decode_q8_0_rows(block * 8, 256)
    assert out.shape == (1, 256)
    assert out[0, :32].tolist() == expected




def test_iq4_xs_hand_constructed_super_block():
    """136-byte blocks: F16 d, u16 scales_h, 4x u8 scales_l, 128 quants.

    Six-bit sub-block scales (4 bits in scales_l nibbles, 2 bits in scales_h
    pairs), dl = d*(ls-32), low nibbles index the first 16 outputs. Layout
    pinned against llama.cpp dequantize_row_iq4_xs / quants.py IQ4_XS.
    """
    from venturi.container.gguf.gguf_quantization import decode_iq4_xs_rows
    ls_values = (32, 33, 31, 40, 48, 0, 63, 12)
    scales_l = bytes(
        (ls_values[2*i] & 0xF) | ((ls_values[2*i+1] & 0xF) << 4) for i in range(4))
    scales_h = sum(((ls >> 4) & 0x3) << (2*ib) for ib, ls in enumerate(ls_values))
    quants = bytes(i | ((15-i) << 4) for i in range(16)) * 8
    block = struct.pack('<eH', 0.5, scales_h) + scales_l + quants
    assert len(block) == 136
    out = decode_iq4_xs_rows(block, 256)
    assert out.shape == (1, 256)
    for ib, ls in enumerate(ls_values):
        dl = 0.5 * (ls - 32)
        chunk = out[0, 32*ib:32*ib+32]
        assert chunk[:16].tolist() == [dl*v for v in LUT]
        assert chunk[16:].tolist() == [dl*v for v in reversed(LUT)]


def test_iq4_xs_rejects_bad_width_payload_and_scale():
    from venturi.container.gguf.gguf_quantization import decode_iq4_xs_rows
    good = struct.pack('<eH', 0.5, 0) + bytes(4) + bytes(128)
    with pytest.raises(ValueError):
        decode_iq4_xs_rows(good, 128)  # width not a multiple of 256
    with pytest.raises(ValueError):
        decode_iq4_xs_rows(good[:135], 256)  # short block
    bad_scale = struct.pack('<e', float('inf')) + bytes(134)
    with pytest.raises(ValueError):
        decode_iq4_xs_rows(bad_scale, 256)  # nonfinite scale refused


def test_iq4_xs_registered_in_row_tables():
    from venturi.container.gguf.gguf_quant_rows import QUANT_TABLES, quant_row_bytes
    assert QUANT_TABLES[23] == ("decode_iq4_xs_rows", 256, 136)
    assert quant_row_bytes(23, 512) == 272
