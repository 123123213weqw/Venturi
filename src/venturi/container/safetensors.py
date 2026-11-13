"""Strict bounded safetensors storage, independent of models and tensor frameworks."""

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from .base import ByteSpan, TensorInfo

DTYPES = {
    "BOOL": ("bool", 1),
    "U8": ("uint8", 1),
    "I8": ("int8", 1),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I16": ("int16", 2),
    "U16": ("uint16", 2),
    "I32": ("int32", 4),
    "U32": ("uint32", 4),
    "I64": ("int64", 8),
    "U64": ("uint64", 8),
}


def _pairs(items):
    out = {}
    for k, v in items:
        if k in out:
            raise ValueError("Duplicate JSON key: " + k)
        out[k] = v
    return out


def _stamp(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class SafetensorsContainer:
    """One standard file. No role directory, format guessing or model policy.

    Tensor offsets are coordinates in the physical data section. Shape/dtype,
    interval coverage and file identity are validated before any tensor read.
    """

    def __init__(self, path, *, max_header_bytes=64 << 20):
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink():
            raise ValueError("Weight source must not be a symlink")
        if type(max_header_bytes) is not int or max_header_bytes < 2:
            raise ValueError("Invalid header budget")
        before = self.path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Weight source must be a regular file")
        self.stamp = _stamp(before)
        with self.path.open("rb") as stream:
            if _stamp(os.fstat(stream.fileno())) != self.stamp:
                raise ValueError("Source changed during open")
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header length")
            length = int.from_bytes(prefix, "little")
            if not 2 <= length <= min(max_header_bytes, before.st_size - 8):
                raise ValueError("Invalid or oversized safetensors header")
            encoded = stream.read(length)
        header = json.loads(
            encoded,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be an object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or any(
            not isinstance(v, str) for v in metadata.values()
        ):
            raise ValueError("Safetensors metadata must contain strings")
        self._metadata = MappingProxyType(metadata)
        self.header_sha256 = hashlib.sha256(encoded).hexdigest()
        self.data_start = 8 + length
        payload = before.st_size - self.data_start
        tensors = {}
        intervals = []
        for name, spec in header.items():
            if not isinstance(name, str) or not name or "\x00" in name:
                raise ValueError("Invalid tensor name")
            if not isinstance(spec, dict) or spec.keys() != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError("Invalid tensor descriptor: " + name)
            if not isinstance(spec["dtype"], str) or spec["dtype"] not in DTYPES:
                raise ValueError("Unsupported safetensors dtype")
            shape = spec["shape"]
            offsets = spec["data_offsets"]
            if (
                not isinstance(shape, list)
                or len(shape) > 16
                or any(type(n) is not int or not 0 <= n < 2**63 for n in shape)
            ):
                raise ValueError("Invalid tensor shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
            ):
                raise ValueError("Invalid tensor offsets")
            a, b = offsets
            dtype, itemsize = DTYPES[spec["dtype"]]
            if not 0 <= a <= b <= payload or math.prod(shape) * itemsize != b - a:
                raise ValueError("Tensor shape/dtype/byte range mismatch: " + name)
            tensors[name] = TensorInfo(
                name, dtype, tuple(shape), self.data_start + a, b - a
            )
            if a != b:
                intervals.append((a, b))
        cursor = 0
        for a, b in sorted(intervals):
            if a != cursor:
                raise ValueError("Tensor ranges overlap or leave holes")
            cursor = b
        if cursor != payload:
            raise ValueError("Unaccounted safetensors data bytes")
        self._tensors = MappingProxyType(tensors)
        self._payload = payload
        self.check_source()

    def check_source(self):
        if self.path.is_symlink() or _stamp(self.path.stat()) != self.stamp:
            raise RuntimeError("Weight source changed; reopen after verification")

    def list_tensors(self):
        return list(self._tensors)

    def tensor_info(self, name):
        return self._tensors[name]

    def metadata(self):
        return dict(self._metadata)

    def logical_offset(self, name):
        return self.tensor_info(name).byte_offset - self.data_start

    def byte_spans(self):
        return [
            ByteSpan(str(self.path), 0, self.data_start, self._payload, self.stamp[2])
        ]

    def total_payload_bytes(self):
        return self._payload

    def read_bytes(self, name, offset=0, size=None, *, max_bytes=64 << 20):
        info = self.tensor_info(name)
        size = info.byte_size - offset if size is None else size
        if (
            any(type(x) is not int for x in (offset, size, max_bytes))
            or offset < 0
            or size < 0
            or offset + size > info.byte_size
            or size > max_bytes
        ):
            raise ValueError("Tensor read exceeds range or budget")
        self.check_source()
        with self.path.open("rb") as stream:
            if _stamp(os.fstat(stream.fileno())) != self.stamp:
                raise RuntimeError("Weight source changed during open")
            stream.seek(info.byte_offset + offset)
            data = stream.read(size)
            if len(data) != size or _stamp(os.fstat(stream.fileno())) != self.stamp:
                raise RuntimeError("Weight source changed during read")
        self.check_source()
        return data

    def signature(self):
        return {
            "path": str(self.path),
            "stamp": self.stamp,
            "header_sha256": self.header_sha256,
        }


class SafetensorsWriter:
    """Bounded per-tensor writes and atomic no-clobber publication.

    Each tensor is written sequentially, but different tensors may interleave.
    Only standard safetensors fields are serialized; no virtual object directory.
    """

    def __init__(self, path, tensors, metadata):
        import tempfile
        import shutil

        self.path = Path(path).expanduser().absolute()
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        if self.path.suffix != ".safetensors":
            raise ValueError("Output must use .safetensors")
        if not self.path.parent.is_dir():
            raise ValueError("Output parent must exist")
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in metadata.items()
        ):
            raise ValueError("Metadata must contain strings")
        header = {"__metadata__": dict(metadata)}
        self.offsets = {}
        self.sizes = {}
        self.next = {}
        cursor = 0
        for name, (dtype, shape) in tensors.items():
            if (
                not isinstance(name, str)
                or name == "__metadata__"
                or not name
                or "\x00" in name
                or not isinstance(dtype, str)
                or dtype not in DTYPES
            ):
                raise ValueError("Invalid tensor name/dtype")
            if (
                not isinstance(shape, (tuple, list))
                or len(shape) > 16
                or any(type(n) is not int or not 0 <= n < 2**63 for n in shape)
            ):
                raise ValueError("Invalid tensor shape")
            size = math.prod(shape) * DTYPES[dtype][1]
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [cursor, cursor + size],
            }
            self.offsets[name] = cursor
            self.sizes[name] = size
            self.next[name] = 0
            cursor += size
        encoded = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        length = (len(encoded) + 7) // 8 * 8
        if length > 64 << 20:
            raise ValueError("Safetensors header budget exceeded")
        self.data_start = 8 + length
        self.total_bytes = self.data_start + cursor
        if shutil.disk_usage(self.path.parent).free < self.total_bytes + (64 << 20):
            raise OSError("Insufficient disk space for output plus reserve")
        self.stream = tempfile.NamedTemporaryFile(
            prefix=".venturi-safetensors-",
            suffix=".partial",
            dir=self.path.parent,
            delete=False,
        )
        self.temporary = Path(self.stream.name)
        self.published = False
        try:
            header_bytes = (
                length.to_bytes(8, "little") + encoded + b" " * (length - len(encoded))
            )
            if self.stream.write(header_bytes) != len(header_bytes):
                raise OSError("Short safetensors header write")
            self.stream.truncate(self.total_bytes)
        except BaseException:
            self.close()
            raise

    def write_slice(self, name, data, offset=0):
        if self.published or self.stream.closed:
            raise ValueError("Writer is closed or already published")
        if (
            type(offset) is not int
            or offset != self.next[name]
            or offset + memoryview(data).nbytes > self.sizes[name]
        ):
            raise ValueError("Out-of-order or oversized tensor write")
        self.stream.seek(self.data_start + self.offsets[name] + offset)
        view = memoryview(data).cast("B")
        while view:
            n = self.stream.write(view)
            if not n:
                raise OSError("Short safetensors write")
            view = view[n:]
        self.next[name] += memoryview(data).nbytes

    def finish(self):
        if self.next != self.sizes:
            raise ValueError("Not all tensor bytes were written")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        SafetensorsContainer(self.temporary)

    def publish(self):
        self.finish()
        os.link(self.temporary, self.path)
        self.published = True

    def close(self):
        self.stream.close()
        self.temporary.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
