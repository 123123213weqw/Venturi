"""Bounded, read-only GGUF v3 header inventory; NOT a weight loader or verifier.

Does not import Torch, read tensor payloads, dequantize, pack or choose a backend.
A complete-length file (or valid partial header) is never a SHA-verified asset.
Array metadata is summarized rather than materialized, notably tokenizer tables.

Layout authority: gguf.h/gguf.cpp at llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d;
IQ/extension block geometry also matches Strata 4b9e05a2a806a3fe24f7b2560d98626c37dade35.
Only the explicit layout subset below is understood; unknown types fail closed.
"""
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
import struct


# (name, elements per block, stored bytes per block). This is layout knowledge,
# not a claim that Venturi implements a decoder/kernel for any of these types.
_LAYOUTS = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18),
    3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24),
    8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36), 10: ('Q2_K', 256, 84),
    11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176),
    14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292),
    16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74),
    18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50),
    20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110),
    22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136),
    24: ('I8', 1, 1), 25: ('I16', 1, 2), 26: ('I32', 1, 4),
    27: ('I64', 1, 8), 28: ('F64', 1, 8), 29: ('IQ1_M', 256, 56),
    30: ('BF16', 1, 2), 39: ('MXFP4', 32, 17),
    43: ('MXFP8', 32, 33), 67: ('MXFP8', 32, 33),
}
_META_FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                 6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
_MAX_I64 = 2**63-1


@dataclass(frozen=True)
class GgufLimits:
    header_bytes: int = 64*2**20
    string_bytes: int = 2**20
    metadata_entries: int = 100_000
    tensor_entries: int = 100_000
    array_items: int = 2_000_000  # aggregate, not per-array
    alignment: int = 2**20

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError('GGUF limits must be positive integers')


@dataclass(frozen=True)
class GgufArraySummary:
    element_type: int
    count: int


@dataclass(frozen=True)
class GgufTensor:
    name: str
    type_id: int
    dtype: str
    dimensions_ggml: tuple[int, ...]  # innermost first, NOT PyTorch axis order
    offset: int                      # absolute byte offset in source file
    bytes: int

    @property
    def shape(self):
        return self.dimensions_ggml[::-1]


@dataclass(frozen=True)
class GgufInventory:
    path: str
    metadata: dict
    tensors: tuple[GgufTensor, ...]
    header_bytes_read: int
    header_sha256: str
    data_offset: int
    expected_file_bytes: int
    observed_file_bytes: int
    file_length_complete: bool
    payload_sha256_verified: bool = False

    @property
    def tensor_bytes(self):
        return sum(t.bytes for t in self.tensors)


class _HeaderReader:
    def __init__(self, stream, available, limits):
        self.stream, self.available, self.limits = stream, available, limits
        self.position, self.array_items = 0, 0
        self.digest = hashlib.sha256()

    def read(self, size):
        if size < 0 or self.position+size > self.limits.header_bytes:
            raise ValueError('GGUF header byte budget exceeded')
        if self.position+size > self.available:
            raise ValueError('Truncated GGUF header, not enough contiguous prefix')
        raw = self.stream.read(size)
        if len(raw) != size:
            raise ValueError('Truncated or changed GGUF header')
        self.digest.update(raw)
        self.position += size
        return raw

    def unpack(self, fmt):
        return struct.unpack('<'+fmt, self.read(struct.calcsize('<'+fmt)))

    def skip(self, size):
        # No large array allocations, seeks beyond EOF or tensor payload reads.
        if self.position+size > min(self.available, self.limits.header_bytes):
            raise ValueError('Truncated GGUF header or header byte budget exceeded')
        while size:
            take = min(size, 64*1024)
            self.read(take)
            size -= take

    def string(self):
        size, = self.unpack('Q')
        if size > self.limits.string_bytes:
            raise ValueError('GGUF string byte budget exceeded')
        try:
            return self.read(size).decode('utf-8', errors='strict')
        except UnicodeDecodeError as e:
            raise ValueError('Invalid UTF-8 in GGUF header string') from e

    def value(self, typ):
        if typ in _META_FORMATS:
            value, = self.unpack(_META_FORMATS[typ])
            if typ == 7:
                if value not in (0, 1):
                    raise ValueError('Invalid GGUF metadata bool')
                return bool(value)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Nonfinite GGUF metadata scalar')
            return value
        if typ == 8:
            return self.string()
        if typ != 9:
            raise ValueError(f'Unsupported GGUF metadata type {typ}')
        element_type, count = self.unpack('IQ')
        self.array_items += count
        if self.array_items > self.limits.array_items:
            raise ValueError('GGUF aggregate array item budget exceeded')
        if element_type in _META_FORMATS:
            self.skip(count*struct.calcsize(_META_FORMATS[element_type]))
        elif element_type == 8:
            for _ in range(count):
                size, = self.unpack('Q')
                if size > self.limits.string_bytes:
                    raise ValueError('GGUF array string byte budget exceeded')
                self.skip(size)
        else:
            raise ValueError('Unsupported/nested GGUF metadata array type')
        return GgufArraySummary(element_type, count)


def inspect_gguf(path, *, expected_bytes=None, allow_partial=False, limits=None):
    """Read only the header of an explicit regular, non-symlinked GGUF v3 file.

    Partial inspection requires its independent expected file size. All tensor
    spans are checked against that size, NOT against the downloaded prefix. The
    caller must still verify a full-file SHA before loading/packing anything.
    Metadata arrays are deliberately unavailable (only type/count summaries).
    Zero-element/scalar tensors and layouts outside our subset are unsupported.
    """
    if type(allow_partial) is not bool:
        raise ValueError('allow_partial must be boolean')
    limits = GgufLimits() if limits is None else limits
    if not isinstance(limits, GgufLimits):
        raise ValueError('Explicit GgufLimits required')
    if expected_bytes is not None and (type(expected_bytes) is not int or not 0 < expected_bytes <= _MAX_I64):
        raise ValueError('Expected file bytes must be a positive int64')
    if allow_partial and expected_bytes is None:
        raise ValueError('Partial GGUF inspection requires expected file bytes')
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Symlinked GGUF file refused')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb', buffering=0) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('GGUF inspection requires a regular file')
        observed = info.st_size
        expected = observed if expected_bytes is None else expected_bytes
        if observed > expected or (not allow_partial and observed != expected):
            raise ValueError('GGUF file length does not match expected bytes')
        reader = _HeaderReader(stream, observed, limits)
        if reader.read(4) != b'GGUF':
            raise ValueError('Not a little-endian GGUF file')
        version, tensor_count, metadata_count = reader.unpack('IQQ')
        if version != 3:
            raise ValueError('Only GGUF version 3 is supported by this inventory')
        if tensor_count > limits.tensor_entries or metadata_count > limits.metadata_entries:
            raise ValueError('GGUF entry count budget exceeded')
        metadata, types = {}, {}
        for _ in range(metadata_count):
            name = reader.string()
            if not name or name in metadata:
                raise ValueError('Empty or duplicate GGUF metadata key')
            typ, = reader.unpack('I')
            metadata[name], types[name] = reader.value(typ), typ
        alignment = metadata.get('general.alignment', 32)
        if ('general.alignment' in types and types['general.alignment'] != 4
                or type(alignment) is not int or alignment < 1 or alignment & (alignment-1)
                or alignment > limits.alignment):
            raise ValueError('GGUF alignment must be bounded uint32 power of two')
        tensors, names, relative_offset = [], set(), 0
        for _ in range(tensor_count):
            name = reader.string()
            if not name or name in names:
                raise ValueError('Empty or duplicate GGUF tensor name')
            names.add(name)
            rank, = reader.unpack('I')
            if not 1 <= rank <= 4:
                raise ValueError('Unsupported GGUF tensor rank (requires 1..4)')
            dimensions = reader.unpack('Q'*rank)
            if any(not 0 < n <= _MAX_I64 for n in dimensions) or math.prod(dimensions) >= _MAX_I64:
                raise ValueError('Invalid/overflowing GGUF tensor dimensions')
            typ, offset = reader.unpack('IQ')
            if typ not in _LAYOUTS:
                raise ValueError(f'Unsupported GGUF tensor layout {typ}')
            dtype, block, size = _LAYOUTS[typ]
            if dimensions[0] % block:
                raise ValueError('GGUF row width is not divisible by quantization block')
            nbytes = math.prod(dimensions)//block*size
            if nbytes > _MAX_I64 or offset != relative_offset:
                raise ValueError('GGUF tensor span overflow or noncanonical/overlapping offset')
            relative_offset += (nbytes+alignment-1)//alignment*alignment
            tensors.append(GgufTensor(name, typ, dtype, dimensions, offset, nbytes))
        header_bytes = reader.position
        data_offset = ((header_bytes+alignment-1)//alignment*alignment
                       if tensor_count else header_bytes)
        if data_offset+relative_offset > expected:
            raise ValueError('GGUF tensor spans exceed expected file bytes')
        tensors = tuple(GgufTensor(t.name, t.type_id, t.dtype, t.dimensions_ggml,
                                  data_offset+t.offset, t.bytes) for t in tensors)
        return GgufInventory(str(path), metadata, tensors, header_bytes,
                             reader.digest.hexdigest(), data_offset, expected,
                             observed, observed == expected)
