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
    super_blocks = len(raw) // 84 * (width // 256) if width >= 256 else len(raw) // row_bytes
    super_blocks = (len(raw) // row_bytes) * (width // 256)
    if super_blocks * 256 > max_values:
        raise ValueError("Q2_K decode exceeds bounded budget")
    values = []
    for b in range(super_blocks):
        base = b * 84
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
    total = super_blocks * 256
    if total % width:
        raise ValueError("Q2_K payload does not compose whole rows")
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(total // width, width)


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
    super_blocks = (len(raw) // row_bytes) * (width // 256)
    if super_blocks * 256 > max_values:
        raise ValueError("Q3_K decode exceeds bounded budget")
    values = []
    for b in range(super_blocks):
        base = b * 110
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
    total = super_blocks * 256
    if total % width:
        raise ValueError("Q3_K payload does not compose whole rows")
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(total // width, width)


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
    super_blocks = (len(raw) // row_bytes) * (width // 256)
    if super_blocks * 256 > max_values:
        raise ValueError("IQ3_XXS decode exceeds bounded budget")
    grid = _iq3_xxs_grid()
    signs_table = _IQ_KSIGNS
    values = []
    for b in range(super_blocks):
        base = b * 98
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
    total = super_blocks * 256
    if total % width:
        raise ValueError("IQ3_XXS payload does not compose whole rows")
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(total // width, width)


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


def _torch():
    import torch
    return torch


def decode_q8_0_rows(raw, width, *, max_values=2**20):
    """Q8_0 rows: 32-value blocks of 34 bytes (F16 scale + 32 int8 codes)."""
    torch = _torch()
    if type(width) is not int or width < 32 or width > 2**16 or width % 32:
        raise ValueError("Q8_0 width must be a multiple of 32 in [32, 65536]")
    row_bytes = width // 32 * 34
    if not isinstance(raw, bytes) or len(raw) % row_bytes:
        raise ValueError("Q8_0 row payload must be complete 34-byte blocks")
    blocks = len(raw) // 34
    if blocks * 32 > max_values * 32 // width * width and blocks * 32 > max_values:
        raise ValueError("Q8_0 decode exceeds bounded budget")
    values = []
    for b in range(blocks):
        base = b * 34
        d = struct.unpack_from("<e", raw, base)[0]
        if not math.isfinite(d):
            raise ValueError("Q8_0 nonfinite block scale")
        codes = struct.unpack_from("<32b", raw, base + 2)
        values.extend(d * code for code in codes)
    total = blocks * 32
    if total % width:
        raise ValueError("Q8_0 payload does not compose whole rows")
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(total // width, width)


def decode_q6_k_rows(raw, width, *, max_values=2**20):
    """Q6_K rows: 256-value super-blocks of 210 bytes (GGUF type 14).

    Direct torch transcription of the pinned reference reshape chain:
    ql[128] + qh[64] + 16 int8 scales + one F16 d; 6-bit codes assemble as
    (ql nibble | qh 2-bit << 4) - 32 per group of 32 values.
    """
    torch = _torch()
    if type(width) is not int or width < 256 or width > 2**16 or width % 256:
        raise ValueError("Q6_K width must be a multiple of 256 in [256, 65536]")
    row_bytes = width // 256 * 210
    if not isinstance(raw, bytes) or len(raw) % row_bytes:
        raise ValueError("Q6_K row payload must be complete 210-byte blocks")
    blocks = (len(raw) // row_bytes) * (width // 256)
    if blocks * 256 > max_values:
        raise ValueError("Q6_K decode exceeds bounded budget")
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(blocks, 210)
    ql, qh, scales_raw, d_raw = data.split([128, 64, 16, 2], dim=1)
    import numpy as np
    d = torch.from_numpy(
        np.frombuffer(d_raw.reshape(-1, 2).numpy().tobytes(), dtype="<e").astype(np.float32)
    ).reshape(blocks, 1)
    if not bool(torch.isfinite(d).all()):
        raise ValueError("Q6_K nonfinite super-block scales")
    scales = scales_raw.view(torch.int8).to(torch.float32).reshape(blocks, 16)  # per super-block
    # Reference chain, transcribed exactly:
    #   ql.reshape(n,-1,1,64) >> [0,4] -> (n,2,2,64) -> reshape(n,-1,32) = (n,8,32)
    #   qh.reshape(n,-1,1,32) >> [0,2,4,6] -> (n,2,4,32) -> reshape(n,-1,32) = (n,8,32)
    ql = ql.reshape(blocks, 2, 1, 64) >> torch.tensor([0, 4], dtype=torch.uint8).view(1, 1, 2, 1)
    ql = (ql & 0x0F).reshape(blocks, 8, 32)
    qh = qh.reshape(blocks, 2, 1, 32) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8).view(1, 1, 4, 1)
    qh = (qh & 0x03).reshape(blocks, 8, 32)
    q = (ql | (qh << 4)).view(torch.int8) - 32
    q = q.reshape(blocks, 16, 16).to(torch.float32)  # pinned reshape (n, QK_K//16, -1)
    dl = (d * scales).reshape(blocks, 16, 1)
    total = blocks * 256
    if total % width:
        raise ValueError("Q6_K payload does not compose whole rows")
    return (dl * q).reshape(total // width, width).clone()


def decode_iq2_s_rows(raw, width, *, max_values=2**20):
    """IQ2_S rows: 256-value super-blocks of 82 bytes (GGUF type 22).

    Layout: F16 d, 32 code bytes (8-bit grid index into the 1024x8 grid with
    a 2-bit high extension from qh), 32 sign bytes, 8 qh bytes, 8 packed
    scales (two 4-bit per byte). Transcribed from the pinned reference.
    """
    torch = _torch()
    if type(width) is not int or width < 256 or width > 2**16 or width % 256:
        raise ValueError("IQ2_S width must be a multiple of 256 in [256, 65536]")
    row_bytes = width // 256 * 82
    if not isinstance(raw, bytes) or len(raw) % row_bytes:
        raise ValueError("IQ2_S row payload must be complete 82-byte blocks")
    blocks = (len(raw) // row_bytes) * (width // 256)
    if blocks * 256 > max_values:
        raise ValueError("IQ2_S decode exceeds bounded budget")
    grid = _iq2_s_grid()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(blocks, 82)
    d_raw, qs, signs, qh, scales_raw = data.split([2, 32, 32, 8, 8], dim=1)
    import numpy as np
    d = torch.from_numpy(
        np.frombuffer(d_raw.reshape(-1, 2).numpy().tobytes(), dtype="<e").astype(np.float32)
    ).reshape(blocks, 1)
    if not bool(torch.isfinite(d).all()):
        raise ValueError("IQ2_S nonfinite super-block scale")
    scales = scales_raw.reshape(blocks, -1, 1) >> torch.tensor([0, 4], dtype=torch.uint8).view(1, 1, 2)
    scales = (scales & 0x0F).reshape(blocks, -1)
    db = (d * (0.5 + scales.to(torch.float32)) * 0.25).reshape(blocks, -1, 1, 1)
    signs = (signs.reshape(blocks, -1, 1) >> torch.arange(8, dtype=torch.uint8).view(1, 1, 8)) & 1
    signs = torch.where(signs == 0, 1.0, -1.0).reshape(blocks, -1, 2, 8)
    qh_bits = (qh.reshape(blocks, -1, 1) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8).view(1, 1, 4)) & 0x03
    qs16 = qs.to(torch.int64).reshape(blocks, -1) | (qh_bits.to(torch.int64).reshape(blocks, -1) << 8)
    rows = torch.tensor(grid, dtype=torch.float32)
    grid_vals = rows[qs16.reshape(-1)].reshape(blocks, -1, 2, 8)
    total = blocks * 256
    if total % width:
        raise ValueError("payload does not compose whole rows")
    return (db * grid_vals * signs).reshape(total // width, width).clone()


def decode_iq3_s_rows(raw, width, *, max_values=2**20):
    """IQ3_S rows: 256-value super-blocks of 110 bytes (GGUF type 21).

    Layout: F16 d, 64 code bytes (8-bit index into the 512x4 grid with a
    1-bit high extension), 8 qh bytes, 32 sign bytes, 8 packed scales
    (two 4-bit per byte; db = d * (1 + 2 * scale)).
    """
    torch = _torch()
    if type(width) is not int or width < 256 or width > 2**16 or width % 256:
        raise ValueError("IQ3_S width must be a multiple of 256 in [256, 65536]")
    row_bytes = width // 256 * 110
    if not isinstance(raw, bytes) or len(raw) % row_bytes:
        raise ValueError("IQ3_S row payload must be complete 110-byte blocks")
    blocks = (len(raw) // row_bytes) * (width // 256)
    if blocks * 256 > max_values:
        raise ValueError("IQ3_S decode exceeds bounded budget")
    grid = _iq3_s_grid()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(blocks, 110)
    d_raw, qs, qh, signs, scales_raw = data.split([2, 64, 8, 32, 4], dim=1)
    import numpy as np
    d = torch.from_numpy(
        np.frombuffer(d_raw.reshape(-1, 2).numpy().tobytes(), dtype="<e").astype(np.float32)
    ).reshape(blocks, 1)
    if not bool(torch.isfinite(d).all()):
        raise ValueError("IQ3_S nonfinite super-block scale")
    scales = scales_raw.reshape(blocks, -1, 1) >> torch.tensor([0, 4], dtype=torch.uint8).view(1, 1, 2)
    scales = (scales & 0x0F).reshape(blocks, -1)
    db = (d * (1 + 2 * scales.to(torch.float32))).reshape(blocks, -1, 1, 1)
    signs = (signs.reshape(blocks, -1, 1) >> torch.arange(8, dtype=torch.uint8).view(1, 1, 8)) & 1
    signs = torch.where(signs == 0, 1.0, -1.0).reshape(blocks, -1, 4, 8)
    qh_bits = (qh.reshape(blocks, -1, 1) >> torch.arange(8, dtype=torch.uint8).view(1, 1, 8)) & 1
    qs16 = qs.to(torch.int64).reshape(blocks, -1) | (qh_bits.to(torch.int64).reshape(blocks, -1) << 8)
    rows = torch.tensor(grid, dtype=torch.float32)
    grid_vals = rows[qs16.reshape(-1)].reshape(blocks, -1, 4, 8)
    total = blocks * 256
    if total % width:
        raise ValueError("payload does not compose whole rows")
    return (db * grid_vals * signs).reshape(total // width, width).clone()


# IQ4_XS (type 23): deliberately unsupported. The pinned numpy reference
# has a non-conserving reshape chain (64 nibble bytes cannot cover its declared
# 256 values), so no trustworthy semantics exist to transcribe; only three
# stray tensors in the wild use it.



# Pinned lookup grids for IQ2_S (1024x8) and IQ3_S (512x4), extracted verbatim
# from llama.cpp 3cf03257 quants.py (sha256 2c927a1b...).
_IQ2_S_GRID_HEX = b"00000200050008000a0011001400160019002000220025002800410044004600490050005200550058006100640066006900800082008500880091009400a000a500aa0001010401060109011001120115011801210124014001420145014801510154015601590160016501680181018401900192019501a101a40100020202050208021102140220022a02410244024602490250025502800285028a029402a20201040404060409041004120415041804210424042604290440044204450448044a0451045404560459046004620465048104840486048904900495049804a104a40400050205050508050a0511051405160519052005250528054105440546054905500552055505580561056405800582058505880591059405a00501060406060609061006150640064506480651065406600681068406900600080208050808081108140816081908200825082a084108440846084908500852085508580861086408800885089408aa08010904091009120915091809210940094509480951095409600981099009000a110a140a220a280a2a0a500a990a01100410061009101010121015101810211024102610401042104510481051105410561059106010621065106810811084108610901095109810a110a41000110211051108110a1111111411161119112011221125112811411144114611491150115211551158116111641180118211851188119111941101120412091210121512211224124012451251125412811284129012001402140514081411141414161419142014251428144114441446144914501452145514581461146414801482148514881491149414a014011504150615091510151215151518152115241540154215451548155115541560158115841590150016051608161116141620164116441650168016aa160118041806180918101815181818211840184218451848185118541860188118841800190219051908191119141920194119441950196919a219041a101a401a561a00200220052008201120142016201920202025202a2041204420502052205520642080208a209420aa2001210421102112211521212140214221452151215421602181218421902100220a22222228222a224422502288228a22a8220124042406240924102415241824212424244024422445244824512454246024812484249024002505250825112514252025412544255025662580250126042610264026592600280528112814284128442850288a28aa2801290429102995290a2a222a642a882a8a2a014004400640094010401240154018401a4021402440264040404240454048404a40514054405640594060406240654081408440904095409840a140a440004102410541084111411441164119412041224125414141444146414941504152415541584161416441804182418541884191419441a04101420442104212421542184224424042454248425142544260428142844200440244054408440a44114414441644194420442244254428444144444446444944504452445544584461446444804482448544884491449444a044014504450645094510451245154518452145244540454245454548455145544560456a4581458445904500460246054608461146144620464146444650468046a546014804480948104812481548184821482448404842484548484851485448604884489048004902490549084911491449204941494449504980499649014a044a104a404a0050025005500850115014501650195020502250255028504150445046504950505052505550585061506450805082508550885091509450015104510651095110511251155118512151245140514251455148515151545160518151845190510052055208521152145220524152445250526952805201540454065409541054125415541854215424544054425445544854515454546054815484549054005502550555085511551455205541554455505580550156045610562656405600580258055808581158145820584158445850585a5880580159045910594059005a195a855aa85a016004600660106012601560186021602460406045604860516054606060846090600061026105610861116114612061416144615061806199610462106240625662a162006405640864116414642064416444645064806401650465106540654a6568659265006694660168046810686568986800692a69426aa16a00800280058008801180148019802080258041804480508052805580588061808080858091809480018104810981108112811581188121812481408142814581488151815481818184819081a981008205820a82118214824182448250820184048406840984108412841584188421844084428445844884518454846084818484849084008502850585088511851485208541854485508580858a85018604861086298640860088058811881488418844885088a2880189048940896589228a588a5a8a828aa28a01900490099010901290159018902490409042904590489051905490609081908490909000910591119114914191449150915a910192049210924092a69200940294059408941194149420944194449450948094969401950495109540959895a19500964696649601980498109826984098a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa040a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"

_IQ2_S_GRID_MAP = (0x08, 0x19, 0x2B)

_IQ3_S_GRID_HEX = b"00000100020005000700100011001200140016002000210025003300400042004500470051005300600062007100740077000001010102010401100111011501200123012701310135014401610165017201000201020502070210021302160221022502300234024202450247025102530270027302030311031503200322033103330336034403500352036703710375030004130417042104240432044004430451047004020504052005220526053305410545054705660573050606110613063106520671060007020704072007220726073307500754070010011002100410101011101310151017102010221031103410361054105610611072100011011103110611101114112111301133114111501152117011761100121212151217122012241232124012431255126012721201130413071310131313211327133013341341136213701303140514121414143114331442144614501454140115101513152115301532155115201624162716441646160117031710171217211735174117621770170020012003200520072010201220142016202120232027203020322041204320452050205220672070207320752000210221102113211721222125213121342142215121012204220722212223223022372241225322572271227422002302230523112322232423312333234223502366230124072420242324322435244124722475240425112522253725402553257025002602260726212655266126052711272627302743275027023011301330153017302230313033303530423044304730513063307130013103310531143121312331403160317231763100321232203232323432503201331033143321332333273330334133433347335533733303341134163422343134523460346434013510351235253532354435563573351636413601370337203722373537004004401240204024402740324041405040704002410741114113412241304135414341514155410142034210421542214233424042574262427042044311431343204322433143354300440244244437444044714405450745214562451346344660461047154730474347514702501050145022504050445047505250665074500151035105511251215132517251005211522352305236525352025307531053275344535153655373530154045420543254465412552655515553554256025704572257116013601560316033606060006120612761646112623462426255626262706200631463216340632564436462640065036534656065056640661167136700700470077020702270367040705470627002711171247143714571017204721072167221723072517202733273357353730174057413742074507422754275027631760077"

_IQ3_S_GRID_MAP = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F)


def _hex_rows(grid_hex, grid_map, rows, elems):
    """Decode packed hex grid: ceil(log2(len(map))) bits per element."""
    import math
    bits = (len(grid_map) - 1).bit_length()
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    shifts = list(range(0, 8, 8 // per_byte))  # pinned step: 8 // elems_per_byte
    values = []
    for i in range(0, len(grid_hex), 2):
        high = grid_hex[i] + 9 if grid_hex[i] > 0x40 else grid_hex[i]
        low = grid_hex[i + 1] + 9 if grid_hex[i + 1] > 0x40 else grid_hex[i + 1]
        byte = (high & 0x0F) << 4 | (low & 0x0F)
        for shift in shifts:
            values.append(float(grid_map[(byte >> shift) & mask]))
    return [tuple(values[r * elems:(r + 1) * elems]) for r in range(rows)]


def _iq2_s_grid():
    return _hex_rows(_IQ2_S_GRID_HEX, _IQ2_S_GRID_MAP, 1024, 8)


def _iq3_s_grid():
    return _hex_rows(_IQ3_S_GRID_HEX, _IQ3_S_GRID_MAP, 512, 4)

