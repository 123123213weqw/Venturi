"""Bounded CPU GGUF row decoding oracles; not checkpoint admission or a GPU op.

IQ4_NL is a nonlinear codebook format, NOT uniform signed int4 and NOT the
Flash-Next Q4 layout. Caller must separately bind authenticated tensor geometry
and source bytes. This module does not open/mmap files or load a whole table.
"""
import math
import struct


_IQ4_NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10,
                  1, 13, 25, 38, 53, 69, 89, 113)


def decode_iq4_nl_rows(raw, width, *, max_values=2**20):
    """Decode immutable packed rows into private CPU float32 [rows,width].

    Each 32-value block is a little-endian F16 scale plus 16 nibble-pair bytes:
    LOW nibbles form the first 16 columns; HIGH nibbles form the next 16.
    Negative, signed-zero and subnormal finite scales are valid; NaN/Inf scales
    are refused. This scalar oracle is not a tuned production CPU/GPU kernel.
    """
    if type(width) is not int or not 32 <= width <= 2**16 or width % 32:
        raise ValueError('IQ4_NL row width must be bounded positive blocks of 32')
    if type(max_values) is not int or not 0 < max_values <= 2**20:
        raise ValueError('Explicit bounded output value budget required')
    if not isinstance(raw, bytes):
        raise ValueError('Immutable packed IQ4_NL bytes required')
    row_bytes = (width//32)*18
    rows, remainder = divmod(len(raw), row_bytes)
    if remainder or not rows or rows*width > max_values:
        raise ValueError('Complete nonempty IQ4_NL rows within value budget required')
    # Validate every scale before allocating output; never silently clean NaNs.
    scales = tuple(struct.unpack_from('<e', raw, offset)[0] for offset in range(0, len(raw), 18))
    if not all(math.isfinite(value) for value in scales):
        raise ValueError('Nonfinite IQ4_NL block scale')
    values = []
    for block, scale in enumerate(scales):
        packed = raw[block*18+2:block*18+18]
        values.extend(scale*_IQ4_NL_VALUES[value & 15] for value in packed)
        values.extend(scale*_IQ4_NL_VALUES[value >> 4] for value in packed)
    import torch
    return torch.tensor(values, dtype=torch.float32, device='cpu').reshape(rows, width)


def decode_q5_k_rows(raw, width, *, max_values=2**20):
    """Scalar CPU Q5_K [rows,width] oracle, 256 values / 176-byte block.

    Two F16 scales, twelve packed 6-bit scale/min bytes, 32 high-bit bytes
    and 128 low-nibble bytes. This is affine Q5_K, not IQ4_NL or uniform Q5.
    All arithmetic/output is F32 semantics; no fused matmul or GPU emulation.
    """
    if type(width) is not int or not 256 <= width <= 2**16 or width%256:
        raise ValueError('Q5_K row width must be bounded positive blocks of 256')
    if type(max_values) is not int or not 0 < max_values <= 2**20:
        raise ValueError('Explicit bounded output value budget required')
    if not isinstance(raw,bytes):raise ValueError('Immutable packed Q5_K bytes required')
    rows,remainder=divmod(len(raw),(width//256)*176)
    if remainder or not rows or rows*width>max_values:
        raise ValueError('Complete nonempty Q5_K rows within value budget required')
    scales=tuple(struct.unpack_from('<ee',raw,offset) for offset in range(0,len(raw),176))
    if any(not math.isfinite(v) for pair in scales for v in pair):
        raise ValueError('Nonfinite Q5_K block scale/min')
    values=[]
    for block,(d,dmin) in enumerate(scales):
        offset=block*176;packed=raw[offset+4:offset+16]
        qh=raw[offset+16:offset+48];ql=raw[offset+48:offset+176]
        for group in range(8):
            if group<4:scale=packed[group]&63;minimum=packed[group+4]&63
            else:
                i=group-4
                scale=(packed[i+8]&15)|((packed[i]>>6)<<4)
                minimum=(packed[i+8]>>4)|((packed[i+4]>>6)<<4)
            # Products of finite F16 values and these small integers are exact
            # in F32. Final subtraction rounds once when creating F32 output.
            ds=d*scale;dm=dmin*minimum
            for column in range(32):
                low=(ql[(group//2)*32+column]>>(4*(group%2)))&15
                high=(qh[column]>>group)&1
                values.append(ds*(low|(high<<4))-dm)
    import torch
    return torch.tensor(values,dtype=torch.float32,device='cpu').reshape(rows,width)


def decode_bf16_rows(raw, width, *, max_values=2**20):
    """Little-endian BF16 -> private CPU F32 rows by exact bit expansion.

    Preserve signed zero/subnormal bit patterns without numeric conversion or
    changing process-wide denormal settings. NaN/Inf weights are refused.
    """
    if type(width) is not int or not 1<=width<=2**16:
        raise ValueError('Bounded positive BF16 row width required')
    if type(max_values) is not int or not 0<max_values<=2**20:
        raise ValueError('Explicit bounded output value budget required')
    if not isinstance(raw,bytes):raise ValueError('Immutable packed BF16 bytes required')
    rows,remainder=divmod(len(raw),width*2)
    if remainder or not rows or rows*width>max_values:
        raise ValueError('Complete nonempty BF16 rows within value budget required')
    words=tuple(value[0] for value in struct.iter_unpack('<H',raw))
    if any(word&0x7f80==0x7f80 for word in words):
        raise ValueError('Nonfinite BF16 weight')
    bits=[(word<<16) if word<0x8000 else (word<<16)-2**32 for word in words]
    import torch
    return torch.tensor(bits,dtype=torch.int32,device='cpu').view(torch.float32).reshape(rows,width)
