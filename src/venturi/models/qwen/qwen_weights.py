"""Qwen role admission over standard safetensors tensor names, not a container ABI."""

from dataclasses import dataclass
import json
from types import MappingProxyType
import torch
from venturi.container.safetensors import SafetensorsContainer
from venturi.quantization.codecs import codec_for
from venturi.ops.weights import DenseWeight, Q8RowSplitWeight
from venturi.models.qwen.qwen_source import QwenSourcePlan

FLOAT_DTYPES = {
    "fp16": ("float16", torch.float16),
    "bf16": ("bfloat16", torch.bfloat16),
    "fp32": ("float32", torch.float32),
}


@dataclass(frozen=True)
class QwenTensor:
    role: str
    source: str
    shape: tuple
    format: str
    planes: tuple

    @property
    def nbytes(self):
        return sum(p.byte_size for p in self.planes)


@dataclass(frozen=True)
class QwenWeights:
    source: SafetensorsContainer
    plan: object
    tensors: object
    provenance: object
    component: str = "text"

    @classmethod
    def open(cls, path):
        source = SafetensorsContainer(path)
        meta = source.metadata()
        if meta.get("venturi.component") != "text":
            raise ValueError("Expected a Qwen text checkpoint")
        if not {"venturi.config", "venturi.matrix_format"} <= meta.keys():
            raise ValueError("Missing Qwen checkpoint metadata")
        cfg = json.loads(meta["venturi.config"])
        fmt = meta["venturi.matrix_format"]
        plan = QwenSourcePlan.from_config(cfg, matrix_format=fmt)
        return cls._admit(source, plan, "text")

    @classmethod
    def _admit(cls, source, plan, component):
        available = set(source.list_tensors())
        expected = set()
        tensors = {}
        for spec in plan.bindings:
            codec = codec_for(spec.format)
            planes = []
            if codec.quantized:
                n, k = spec.shape
                groups = ((k + 127) // 128 * 128) // codec.group
                want = [
                    (
                        spec.source + ".codes",
                        (n, groups * 32),
                        "int8" if codec.bits == 8 else "uint8",
                    )
                ]
                if codec.high_plane:
                    want.append(
                        (
                            spec.source + ".high",
                            (n, groups * 8 * (codec.bits - 4)),
                            "uint8",
                        )
                    )
                want.append((spec.source + ".scales", (n, groups), "float16"))
            else:
                want = [(spec.source, spec.shape, FLOAT_DTYPES[spec.format][0])]
            for name, shape, dtype in want:
                if name not in available:
                    raise ValueError("Missing Qwen tensor: " + name)
                info = source.tensor_info(name)
                if info.shape != tuple(shape) or info.dtype != dtype:
                    raise ValueError("Qwen tensor shape/dtype mismatch: " + name)
                if not codec.quantized and info.byte_size > 8 << 20 and len(shape) != 2:
                    raise ValueError("Small parameter read bound exceeded")
                planes.append(info)
                expected.add(name)
            tensors[spec.binding] = QwenTensor(
                spec.binding, spec.source, tuple(spec.shape), spec.format, tuple(planes)
            )
        if available != expected:
            raise ValueError(
                "Unexpected Qwen tensors: " + str(sorted(available - expected)[:5])
            )
        if plan.tied_embeddings:
            tensors["text/output_head"] = tensors["text/token_embedding"]
        provenance = json.loads(source.metadata().get("venturi.source", "null"))
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("Invalid Qwen source provenance")
        return cls(source, plan, MappingProxyType(tensors), provenance, component)

    @property
    def config(self):
        return self.plan.config

    def descriptor(self, role):
        if role not in self.tensors:
            raise ValueError("Unknown Qwen role: " + role)
        return self.tensors[role]

    def plane_offset(self, role, plane="codes"):
        desc = self.descriptor(role)
        name = desc.source + ("." + plane if codec_for(desc.format).quantized else "")
        if name not in {p.name for p in desc.planes}:
            raise ValueError("Unknown matrix plane")
        return self.source.logical_offset(name)

    def layer_bindings(self, layer):
        if type(layer) is not int or not 0 <= layer < self.config["num_hidden_layers"]:
            raise ValueError("Layer index out of range")
        prefix = f"text/layers/{layer}/"
        return tuple(
            s.binding for s in self.plan.bindings if s.binding.startswith(prefix)
        )

    def _read(self, info, *, max_bytes=8 << 20):
        data = self.source.read_bytes(info.name, max_bytes=max_bytes)
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "int8": torch.int8,
            "uint8": torch.uint8,
        }[info.dtype]
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(info.shape)

    def read_float(self, role):
        desc = self.descriptor(role)
        if codec_for(desc.format).quantized:
            raise ValueError("Floating read never dequantizes a matrix")
        value = self._read(desc.planes[0])
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite Qwen parameter: " + role)
        return value

    def read_plane(self, role, plane):
        desc = self.descriptor(role)
        codec = codec_for(desc.format)
        labels = (
            ("codes", "high", "scales") if codec.high_plane else ("codes", "scales")
        )
        if not codec.quantized or plane not in labels:
            raise ValueError("Unknown quantized plane")
        info = desc.planes[labels.index(plane)]
        # One owned plane; bounded physical reads avoid a second multi-GB bytes copy.
        dtype = {"int8": torch.int8, "uint8": torch.uint8, "float16": torch.float16}[
            info.dtype
        ]
        out = torch.empty(info.shape, dtype=dtype)
        raw = out.view(torch.uint8).view(-1)
        for start in range(0, info.byte_size, 16 << 20):
            size = min(16 << 20, info.byte_size - start)
            raw[start : start + size].copy_(
                torch.frombuffer(
                    bytearray(
                        self.source.read_bytes(
                            info.name, start, size, max_bytes=16 << 20
                        )
                    ),
                    dtype=torch.uint8,
                )
            )
        return out

    def read_linear(self, role):
        desc = self.descriptor(role)
        codec = codec_for(desc.format)
        if codec.bits == 8:
            codes = self.read_plane(role, "codes")
            scales = self.read_plane(role, "scales")
            if (
                bool((codes == -128).any())
                or not bool(torch.isfinite(scales).all())
                or bool((scales < 0).any())
            ):
                raise ValueError("Invalid Q8 stored values")
            return Q8RowSplitWeight(codes, scales, desc.shape[1])
        if codec.quantized:
            from venturi.quantization.codecs import dequantize_planes

            codes = self.read_plane(role, "codes")
            scales = self.read_plane(role, "scales")
            high = self.read_plane(role, "high") if codec.high_plane else None
            return DenseWeight(
                dequantize_planes(codes, scales, high, desc.shape, desc.format)
            )
        info = desc.planes[0]
        value = self._read(info, max_bytes=info.byte_size)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite dense Qwen matrix")
        return DenseWeight(value)

    def signature(self):
        return {
            "source": self.source.signature(),
            "config": self.config,
            "component": self.component,
            "tensors": [
                (
                    r,
                    d.source,
                    d.shape,
                    d.format,
                    tuple(
                        (p.name, p.dtype, p.shape, p.byte_offset, p.byte_size)
                        for p in d.planes
                    ),
                )
                for r, d in sorted(self.tensors.items())
            ],
        }
