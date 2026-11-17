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


def decode_q2_k_rows(raw, width, *, max_values=2**20):
    """Q2_K rows: 256-value super-blocks of 84 bytes (GGUF type 10).

    Layout per super-block: 16 scale bytes (low nibble scale factor, high
    nibble min factor), 64 two-bit quant bytes, one F16 d and one F16 dmin.
    Value (g, i) reads quant byte (g//8)*32 + (g%2)*16 + i at bit shift
    2*((g%8)//2); products of finite F16 scales and small integers are exact
    in F32, with one rounding on the final subtraction.
    """
    import torch
    if type(width) is not int or width < 256 or width > 2**16 or width % 256:
        raise ValueError("Q2_K width must be a multiple of 256 in [256, 65536]")
    row_bytes = width // 256 * 84
    if not isinstance(raw, bytes) or len(raw) % row_bytes:
        raise ValueError("Q2_K row payload must be complete 84-byte blocks")
    blocks = len(raw) // row_bytes
    if blocks * width > max_values:
        raise ValueError("Q2_K decode exceeds bounded budget")
    values = []
    for b in range(blocks):
        base = b * row_bytes
        d = struct.unpack_from("<e", raw, base + 80)[0]
        dmin = struct.unpack_from("<e", raw, base + 82)[0]
        if not (math.isfinite(d) and math.isfinite(dmin)):
            raise ValueError("Q2_K nonfinite super-block scales")
        scales, qs = raw[base:base + 16], raw[base + 16:base + 80]
        for g in range(16):
            dl = d * (scales[g] & 0xF)
            ml = dmin * (scales[g] >> 4)
            for i in range(16):
                byte = qs[(g // 8) * 32 + (g % 2) * 16 + i]
                values.append(dl * ((byte >> (2 * ((g % 8) // 2))) & 3) - ml)
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(blocks, width)


def decode_q3_k_rows(raw, width, *, max_values=2**20):
    """Q3_K rows: 256-value super-blocks of 110 bytes (GGUF type 11).

    Layout per super-block: 32 high-bit mask bytes, 64 two-bit quant bytes,
    12 packed 6-bit super-block scales, one F16 d. Scales unpack to signed
    6-bit (raw minus 32). The mask bit is XORed: a set bit offsets the quant
    by zero, a clear bit by -4. Quant byte/shift selection matches Q2_K.
    """
    import torch
    if type(width) is not int or width < 256 or width > 2**16 or width % 256:
        raise ValueError("Q3_K width must be a multiple of 256 in [256, 65536]")
    row_bytes = width // 256 * 110
    if not isinstance(raw, bytes) or len(raw) % row_bytes:
        raise ValueError("Q3_K row payload must be complete 110-byte blocks")
    blocks = len(raw) // row_bytes
    if blocks * width > max_values:
        raise ValueError("Q3_K decode exceeds bounded budget")
    values = []
    for b in range(blocks):
        base = b * row_bytes
        hmask = raw[base:base + 32]
        qs = raw[base + 32:base + 96]
        packed = raw[base + 96:base + 108]
        d = struct.unpack_from("<e", raw, base + 108)[0]
        if not math.isfinite(d):
            raise ValueError("Q3_K nonfinite super-block scale")
        for g in range(16):
            low = (packed[g % 8] >> (4 * (g // 8))) & 0x0F
            high = (packed[8 + g % 4] >> (2 * (g // 4))) & 0x03
            dl = d * (((low | (high << 4)) & 0x3F) - 32)
            for i in range(16):
                byte = qs[(g // 8) * 32 + (g % 2) * 16 + i]
                quant = (byte >> (2 * ((g % 8) // 2))) & 3
                column = (g % 2) * 16 + i
                bit = ((hmask[column] >> (g // 2)) & 1) ^ 1
                values.append(dl * (quant - (bit << 2)))
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(blocks, width)


def decode_iq3_xxs_rows(raw, width, *, max_values=2**20):
    """IQ3_XXS rows: 256-value super-blocks of 98 bytes (GGUF type 18).

    Layout per super-block: one F16 d, 64 codebook index bytes (each selects
    a 4-value row of the pinned 256x4 grid), and 32 bytes of little-endian
    uint32 words viewed as EIGHT words. Word m owns value segment m (32
    values): its top 4 bits are the segment gain, its low 28 bits are four
    7-bit sign-table indices (shifts 0, 7, 14, 21), each index yielding eight
    sign bits (bit set means negative). Values flatten (m, k4, lane) with
    strides 32/8/1: codebook byte q covers values m*32 + (q%8//2)*8 + (q%2)*4
    + e for e in 0..3, with sign index (q%8)//2 and lane bit (q%2)*4 + e.
    """
    import torch
    if type(width) is not int or width < 256 or width > 2**16 or width % 256:
        raise ValueError("IQ3_XXS width must be a multiple of 256 in [256, 65536]")
    row_bytes = width // 256 * 98
    if not isinstance(raw, bytes) or len(raw) % row_bytes:
        raise ValueError("IQ3_XXS row payload must be complete 98-byte blocks")
    blocks = len(raw) // row_bytes
    if blocks * width > max_values:
        raise ValueError("IQ3_XXS decode exceeds bounded budget")
    grid = _iq3_xxs_grid()
    signs_table = _IQ_KSIGNS
    values = []
    for b in range(blocks):
        base = b * row_bytes
        d = struct.unpack_from("<e", raw, base)[0]
        qs = raw[base + 2:base + 66]
        words = struct.unpack_from("<8I", raw, base + 66)
        if not math.isfinite(d):
            raise ValueError("IQ3_XXS nonfinite super-block scale")
        gains, sign_bits = [], []
        for word in words:
            gains.append(d * (0.5 + (word >> 28)) * 0.5)
            row = []
            for shift in (0, 7, 14, 21):
                signs = signs_table[(word >> shift) & 0x7F]
                row.extend(-1.0 if signs >> bit & 1 else 1.0 for bit in range(8))
            sign_bits.append(row)
        for q in range(64):
            entries = grid[qs[q]]
            m = q // 8
            lane = (q % 2) * 4
            row = sign_bits[m][(q % 8) // 2 * 8 + lane: (q % 8) // 2 * 8 + lane + 4]
            for e in range(4):
                values.append(gains[m] * entries[e] * row[e])
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(blocks, width)


# Pinned lookup tables from llama.cpp 3cf03257 (gguf-py/gguf/quants.py,
# sha256 2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b):
# the IQ3_XXS grid literals and the shared ksigns bit table.

_IQ3_XXS_GRID_HEX = (
    b"000002000400110013001700200022003100420073007500010103011001120121012501300132014101540170010002020204021102200222023102330237025102570275020103070310031203250370031304370444045704730475040105070532055205350664061007140716074307610701100310101012102110231030103210341047105010001102111111201122110112031210121212211230127212001302132013311346136613011405145014201524154615711505162217401700200220112013202020222026203120422001210321052110211221212130216321672170210022022211221722202222223722402255220123102314237023742335245324032527254125742501270327162745270130103012302130233050306530723000310231"
    b"2031313144314631013203321032253252327232113333333034473472340035063522355535143636366336333760370440174035403740534057407441204237424042604266420743454304445144644425454345704505471047124730471250415070500051065126515551145232527252025353531054235427547254025531555056245742572460446046606460216161611762646230633663446405655265336603672167037005700770107032705270267140711272457252720073157333736073217441740075027524753076"
    b""
)

_IQ_KSIGNS = (
    b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
    b""
)

_IQ3_XXS_GRID_MAP = (0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3E)


def _iq3_xxs_grid():
    """Decode the pinned hex grid into 256 codebook rows of 4 values each.

    Each pair of ASCII hex characters yields one byte; each byte unpacks two
    3-bit grid-map indices, low nibble first, exactly as the pinned reference
    reshapes (byte >> [0, 4]) & 0x07.
    """
    def nibble(character):
        value = character + 9 if character > 0x40 else character
        return value & 0x0F

    indices = []
    for i in range(0, len(_IQ3_XXS_GRID_HEX), 2):
        byte = nibble(_IQ3_XXS_GRID_HEX[i]) << 4 | nibble(_IQ3_XXS_GRID_HEX[i + 1])
        indices.extend((byte & 0x07, (byte >> 4) & 0x07))
    return [
        tuple(float(_IQ3_XXS_GRID_MAP[index]) for index in indices[r * 4:r * 4 + 4])
        for r in range(256)
    ]
