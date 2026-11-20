"""Bounded, CPU-only NInfer v3 artifact inspection and encoded-object access.

This reader preserves stored representations and logical bindings. It does not
cast/dequantize an entire model, choose an execution backend, or replace the
native backend's architecture-specific binder. Layout semantics follow the
external NInfer v3 container and row_split_k128_v1 contracts.
"""

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct


MAGIC = b"NINFER\x00\x03"
PART_MAGIC = b"NINPRT\x00\x03"
MAX_U64 = 2**64 - 1
ROW_FORMATS = {
    "q4_g64_fp16": (4, 64),
    "q5_g64_fp16": (5, 64),
    "q6_g64_fp16": (6, 64),
    "q8_g32_fp16": (8, 32),
}


def _uint(value, label, *, positive=False):
    if type(value) is not int or not int(positive) <= value <= MAX_U64:
        raise ValueError(
            f"Invalid {label}: expected {'positive ' if positive else ''}U64"
        )
    return value


def _identifier(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Invalid artifact identifier")
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def _align(n, alignment=256):
    return (n + alignment - 1) // alignment * alignment


def row_geometry(shape, format):
    n, k = shape
    bits, group = ROW_FORMATS[format]
    groups = _align(k, 128) // group
    base_bytes = n * groups * 32
    high_bytes = n * groups * (8 * (bits - 4) if bits in (5, 6) else 0)
    high_offset = _align(base_bytes)
    scales_offset = high_offset + _align(high_bytes)
    return (
        bits,
        group,
        groups,
        high_offset,
        scales_offset,
        scales_offset + n * groups * 2,
    )


@dataclass(frozen=True)
class PayloadFile:
    path: Path
    logical_start: int
    payload_start: int
    payload_bytes: int


class NInferArtifact:
    def __init__(self, path, *, max_metadata_bytes=64 * 2**20):
        self.path = Path(path).expanduser().resolve(strict=True)
        with self.path.open("rb") as stream:
            header = stream.read(32)
            if len(header) != 32 or header[:8] != MAGIC:
                raise ValueError("Expected a NInfer v3 entry file")
            length = struct.unpack("<Q", header[8:16])[0]
            if not 0 < length <= max_metadata_bytes:
                raise ValueError("Artifact metadata exceeds the configured bound")
            encoded = stream.read(length)
            if len(encoded) != length:
                raise ValueError("Truncated artifact metadata")
            self.manifest = json.loads(
                encoded,
                object_pairs_hook=_unique_pairs,
                parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError(f"Invalid JSON {x}")
                ),
            )
            self.artifact_id = header[16:].hex()
            entry_start = _align(32 + length, 4096)
            if any(stream.read(entry_start - 32 - length)):
                raise ValueError("Nonzero entry alignment padding")
        m = self.manifest
        if (
            not isinstance(m, dict)
            or not {"components", "objects", "bindings", "uses", "files"} <= m.keys()
        ):
            raise ValueError("Missing artifact directory fields")
        if m.keys() - {
            "components",
            "objects",
            "bindings",
            "uses",
            "files",
            "metadata",
            "provenance",
        }:
            raise ValueError("Unknown artifact directory fields")
        if not isinstance(m["files"], list) or not m["files"]:
            raise ValueError("Artifact files must be nonempty")
        self.files, logical, names = [], 0, {self.path.name}
        for index, descriptor in enumerate(m["files"]):
            if not isinstance(descriptor, dict) or descriptor.keys() != {
                "path",
                "payload_bytes",
            }:
                raise ValueError("Invalid file descriptor")
            size = _uint(
                descriptor["payload_bytes"], "file payload_bytes", positive=True
            )
            if index == 0:
                if descriptor["path"] is not None:
                    raise ValueError("Entry file path must be null")
                file, start = self.path, entry_start
            else:
                name = _identifier(descriptor["path"])
                if (
                    name in names
                    or name in (".", "..")
                    or any(c in name for c in ("/", "\\"))
                ):
                    raise ValueError(
                        "Continuation path must be a unique sibling filename"
                    )
                names.add(name)
                file, start = self.path.parent / name, 4096
                with file.open("rb") as stream:
                    part = stream.read(32)
                    if (
                        len(part) != 32
                        or part[:8] != PART_MAGIC
                        or struct.unpack("<Q", part[8:16])[0] != index
                        or part[16:] != header[16:]
                    ):
                        raise ValueError("Continuation header/identity mismatch")
                    if any(stream.read(4096 - 32)):
                        raise ValueError("Nonzero continuation alignment padding")
            if file.stat().st_size != start + size:
                raise ValueError("Artifact file size does not match declared payload")
            self.files.append(PayloadFile(file, logical, start, size))
            logical = _uint(logical + size, "total payload")
        self.payload_bytes = logical
        self.objects, previous_end = {}, 0
        if not isinstance(m["objects"], list) or not m["objects"]:
            raise ValueError("Artifact objects must be nonempty")
        for obj in m["objects"]:
            if not isinstance(obj, dict):
                raise ValueError("Invalid object descriptor")
            name = _identifier(obj.get("id"))
            offset = _uint(obj.get("offset"), "object offset")
            size = _uint(obj.get("bytes"), "object bytes", positive=True)
            if name in self.objects or offset < previous_end or offset + size > logical:
                raise ValueError("Duplicate, overlapping or out-of-bounds object")
            if obj.get("kind") == "resource":
                if (
                    obj.keys() != {"id", "kind", "encoding", "offset", "bytes"}
                    or obj["encoding"] != "raw_bytes_v1"
                ):
                    raise ValueError("Unsupported resource descriptor")
            elif obj.get("kind") == "tensor":
                if obj.keys() != {
                    "id",
                    "kind",
                    "shape",
                    "format",
                    "layout",
                    "offset",
                    "bytes",
                }:
                    raise ValueError("Invalid tensor descriptor")
                shape = obj["shape"]
                if not isinstance(shape, list) or len(shape) > 16 or offset % 256:
                    raise ValueError("Invalid tensor shape/alignment")
                for dimension in shape:
                    _uint(dimension, "tensor dimension", positive=True)
                elements = _uint(
                    math.prod(shape), "tensor element count", positive=True
                )
                format, layout = _identifier(obj["format"]), _identifier(obj["layout"])
                if layout == "contiguous_le_v1" and format in ("bf16", "fp32", "int32"):
                    expected = elements * (2 if format == "bf16" else 4)
                elif (
                    layout == "row_split_k128_v1"
                    and format in ROW_FORMATS
                    and len(shape) == 2
                ):
                    expected = row_geometry(shape, format)[-1]
                else:
                    raise ValueError(
                        f"Unsupported inspection format/layout: {format}/{layout}"
                    )
                if size != expected:
                    raise ValueError("Tensor encoded size does not match layout/shape")
            else:
                raise ValueError("Unknown artifact object kind")
            self.objects[name] = obj
            previous_end = offset + size
        if not isinstance(m["bindings"], dict):
            raise ValueError("Invalid bindings")
        for name, binding in m["bindings"].items():
            _identifier(name)
            self._binding_elements(binding)
        components = m["components"]
        if not isinstance(components, dict) or "text" not in components:
            raise ValueError("Artifact must have a text component")
        for name, component in components.items():
            _identifier(name)
            if not isinstance(component, dict) or not isinstance(
                component.get("config"), dict
            ):
                raise ValueError("Invalid component config")
            if (
                "target" in component
                and _identifier(component["target"]) not in components
            ):
                raise ValueError("Unknown component target")
            resources = component.get("resources", {})
            if not isinstance(resources, dict):
                raise ValueError("Invalid component resources")
            for reference in resources.values():
                if (
                    not isinstance(reference, str)
                    or reference not in self.objects
                    or self.objects[reference]["kind"] != "resource"
                ):
                    raise ValueError("Invalid component resource reference")
        if not isinstance(m["uses"], list):
            raise ValueError("Invalid uses")
        seen = set()
        for use in m["uses"]:
            if (
                not isinstance(use, dict)
                or _identifier(use.get("parameter")) not in m["bindings"]
            ):
                raise ValueError("Unknown use parameter")
            key = (use["parameter"], _identifier(use.get("input")))
            if key in seen or use.get("activation_policy", "A16Only") not in (
                "A16Only",
                "AllowA8",
                "AllowA4",
            ):
                raise ValueError("Duplicate use or unsupported activation policy")
            seen.add(key)
            auxiliaries = use.get("auxiliaries", {})
            if not isinstance(auxiliaries, dict):
                raise ValueError("Invalid use auxiliaries")
            for binding in auxiliaries.values():
                self._binding_elements(binding)

    def _binding_elements(self, binding):
        if not isinstance(binding, dict):
            raise ValueError("Invalid binding")
        if binding.keys() == {"object"}:
            reference = binding["object"]
            if (
                not isinstance(reference, str)
                or reference not in self.objects
                or self.objects[reference]["kind"] != "tensor"
            ):
                raise ValueError("Binding references an unknown tensor")
            return math.prod(self.objects[reference]["shape"])
        if (
            binding.keys() != {"parts"}
            or not isinstance(binding["parts"], list)
            or not binding["parts"]
        ):
            raise ValueError("Invalid binding parts")
        count = 0
        for part in binding["parts"]:
            if not isinstance(part, dict) or part.keys() != {"object", "range"}:
                raise ValueError("Invalid binding part")
            size = self._binding_elements({"object": part["object"]})
            span = part["range"]
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("Invalid binding range")
            begin, end = (_uint(n, "binding range") for n in span)
            if not begin < end <= size:
                raise ValueError("Binding range exceeds logical tensor")
            count = _uint(count + end - begin, "binding size")
        return count

    def read_object(self, name, *, offset=0, size=None, max_bytes=64 * 2**20):
        obj = self.objects[name]
        _uint(offset, "read offset")
        size = obj["bytes"] - offset if size is None else size
        _uint(size, "read size")
        if offset + size > obj["bytes"] or size > max_bytes:
            raise ValueError(
                "Object read exceeds object range or configured byte bound"
            )
        begin, end = obj["offset"] + offset, obj["offset"] + offset + size
        chunks = []
        for file in self.files:
            first, last = (
                max(begin, file.logical_start),
                min(end, file.logical_start + file.payload_bytes),
            )
            if first < last:
                with file.path.open("rb") as stream:
                    stream.seek(file.payload_start + first - file.logical_start)
                    block = stream.read(last - first)
                    if len(block) != last - first:
                        raise ValueError("Truncated object payload")
                    chunks.append(block)
        return b"".join(chunks)

    def decode_row(self, name, row):
        """Diagnostic one-row FP32 reconstruction, never a full-model fallback."""
        obj = self.objects[name]
        if (
            obj.get("layout") != "row_split_k128_v1"
            or obj.get("format") not in ROW_FORMATS
        ):
            raise ValueError(
                "decode_row requires a supported row-split quantized matrix"
            )
        n, k = obj["shape"]
        if type(row) is not int or not 0 <= row < n:
            raise ValueError("Row is out of bounds")
        bits, group, groups, high_offset, scale_offset, _ = row_geometry(
            obj["shape"], obj["format"]
        )
        base = self.read_object(name, offset=row * groups * 32, size=groups * 32)
        high_stride = 8 * (bits - 4) if bits in (5, 6) else 0
        high = self.read_object(
            name,
            offset=high_offset + row * groups * high_stride,
            size=groups * high_stride,
        )
        scales = self.read_object(
            name, offset=scale_offset + row * groups * 2, size=groups * 2
        )
        decoded = []
        for column in range(k):
            g, lane = divmod(column, group)
            if bits == 8:
                word = base[g * 32 + lane]
            else:
                word = (base[g * 32 + lane // 2] >> (4 * (lane % 2))) & 15
                if high_stride:
                    bit = lane * (bits - 4)
                    word |= (
                        (high[g * high_stride + bit // 8] >> (bit % 8))
                        & ((1 << (bits - 4)) - 1)
                    ) << 4
            signed = word - (1 << bits) if word >= 1 << (bits - 1) else word
            if bits == 8 and signed == -128:
                raise ValueError("Q8 code -128 is outside the stored format contract")
            scale = struct.unpack_from("<e", scales, g * 2)[0]
            if not math.isfinite(scale) or scale < 0:
                raise ValueError("Invalid quantization scale")
            decoded.append(signed * scale)
        return decoded

    def summary(self):
        tensors = [o for o in self.objects.values() if o["kind"] == "tensor"]
        return {
            "path": str(self.path),
            "format": "ninfer-v3",
            "artifact_id": self.artifact_id,
            "payload_bytes": self.payload_bytes,
            "file_bytes": sum(f.path.stat().st_size for f in self.files),
            "components": self.manifest["components"],
            "objects": len(self.objects),
            "tensor_count": len(tensors),
            "tensor_bytes": sum(o["bytes"] for o in tensors),
            "formats": dict(Counter(o["format"] for o in tensors)),
            "binding_count": len(self.manifest["bindings"]),
            "scope": "CPU structural inspection and bounded encoded reads, not native materialization or an architecture validation pass",
        }
