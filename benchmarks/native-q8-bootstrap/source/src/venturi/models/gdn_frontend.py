"""Strict small-parameter bindings for the Qwen GDN frontend.

No projections, hidden-state reorder, cache scheduler or full-model fallback.
The same prepared native operators are reusable by other architecture adapters.
"""

from dataclasses import dataclass

import torch

from ..ninfer import NInferArtifact
from ..ops.blocks import _shape


@dataclass(frozen=True)
class GDNFrontendWeights:
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    gated_norm: torch.Tensor
    input_norm: torch.Tensor
    qk_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    source_names: tuple[str, ...]

    @classmethod
    def from_ninfer(cls, artifact, *, layer):
        if not isinstance(artifact, NInferArtifact):
            raise TypeError("Expected validated NInferArtifact")
        cfg = artifact.manifest["components"]["text"]["config"]
        if cfg.get("model_type") != "qwen3_5_text":
            raise ValueError("GDN frontend binding currently requires qwen3_5_text")
        kinds = cfg.get("layer_types", [])
        if (
            type(layer) is not int
            or not 0 <= layer < len(kinds)
            or kinds[layer] != "linear_attention"
        ):
            raise ValueError("Requested layer must be an explicitly declared GDN layer")
        sizes = {}
        for key in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "hidden_size",
        ):
            value = cfg.get(key)
            _shape(value, 1)
            sizes[key] = value
        hq, hv, dk, dv, kernel, hidden = (
            sizes[k]
            for k in (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
                "hidden_size",
            )
        )
        if hv % hq or dk not in (64, 128, 256) or kernel > 32:
            raise ValueError("Unsupported GDN geometry")
        channels = 2 * hq * dk + hv * dv
        _shape(1, channels)
        prefix = f"text/layers/{layer}/"
        wanted = (
            ("gdn/convolution", (kernel, channels), "bf16", torch.bfloat16),
            ("gdn/a_log", (hv,), "fp32", torch.float32),
            ("gdn/dt_bias", (hv,), "fp32", torch.float32),
            ("gdn/norm", (dv,), "bf16", torch.bfloat16),
            ("input_norm", (hidden,), "bf16", torch.bfloat16),
        )
        specs = []
        # Preflight every binding before reading payload bytes.
        for name, shape, format, dtype in wanted:
            binding = artifact.manifest["bindings"].get(prefix + name)
            if not isinstance(binding, dict) or binding.keys() != {"object"}:
                raise ValueError(
                    f"Missing/direct GDN constant binding required: {prefix + name}"
                )
            obj = artifact.objects[binding["object"]]
            if (
                obj["shape"] != list(shape)
                or obj.get("format") != format
                or obj.get("layout") != "contiguous_le_v1"
            ):
                raise ValueError(f"GDN constant shape/format mismatch: {prefix + name}")
            if obj["bytes"] > 8 * 2**20:
                raise ValueError("GDN constant exceeds bounded read size")
            specs.append((obj["id"], shape, dtype))
        weights = []
        for name, shape, dtype in specs:
            raw = bytearray(artifact.read_object(name, max_bytes=8 * 2**20))
            value = torch.frombuffer(raw, dtype=dtype).reshape(shape)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"GDN constant must be finite: {name}")
            weights.append(value)
        return cls(
            *weights, hq, hv, dk, dv, tuple(prefix + name for name, *_ in wanted)
        )

    def upload(self, backend, *, dtype=torch.bfloat16, device="cuda:0", epsilon=1e-6):
        """Explicit activation precision conversion, preserving FP32 gate constants."""
        return ResidentGDNFrontend(
            backend.upload_causal_conv(self.conv.to(dtype), dtype=dtype, device=device),
            backend.upload_gdn_gates(
                self.a_log, self.dt_bias, dtype=dtype, device=device
            ),
            backend.upload_gated_rmsnorm(
                self.gated_norm.to(dtype), dtype=dtype, device=device, epsilon=epsilon
            ),
            backend.upload_zero_centered_rmsnorm(
                self.input_norm, dtype=dtype, device=device, epsilon=epsilon
            ),
        )


@dataclass(frozen=True)
class ResidentGDNFrontend:
    """Resident constants only. State, projections and scheduling remain explicit."""

    conv: object
    gates: object
    gated_norm: object
    input_norm: object

    @property
    def storage_tensors(self):
        return tuple(
            t
            for op in (self.conv, self.gates, self.gated_norm, self.input_norm)
            for t in op.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)
