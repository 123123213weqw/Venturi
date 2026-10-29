"""Explicit text-only equal-axis Qwen3.5 default partial/mRoPE tables.

All three position axes are equal for plain text. Frequency-axis interleaving
therefore reduces to the same 1D phase, while channel rotation remains split-half.
No image/video positions, scaling variants, dynamic update or silent fallback.
"""

from dataclasses import dataclass
import math

import torch

from ..ops.blocks import _shape
from .qwen_gdn_layer import _outside_capture


@dataclass(frozen=True)
class QwenTextRoPE:
    rotary_dim: int
    maximum_position: int
    inverse_frequency: torch.Tensor
    mrope_section: tuple | None

    @classmethod
    def from_config(cls, cfg):
        rope = cfg.get("rope_parameters")
        if not isinstance(rope, dict) or rope.get("rope_type") != "default":
            raise ValueError("Only explicit default Qwen text RoPE is supported")
        d, maximum = cfg.get("head_dim"), cfg.get("max_position_embeddings")
        _shape(d, maximum)
        factor, base = rope.get("partial_rotary_factor", 1.0), rope.get("rope_theta")
        if (
            type(factor) not in (int, float)
            or not math.isfinite(factor)
            or not 0 < factor <= 1
            or type(base) not in (int, float)
            or not math.isfinite(base)
            or base <= 0
        ):
            raise ValueError("Invalid rotary factor/theta")
        dim = int(d * factor)
        if dim < 2 or dim % 2:
            raise ValueError("Rotary width must be positive and even")
        section = rope.get("mrope_section")
        if section is not None and (
            not isinstance(section, (list, tuple))
            or len(section) != 3
            or any(type(x) is not int or x < 0 for x in section)
            or sum(section) != dim // 2
        ):
            raise ValueError("mRoPE sections must partition the rotary half width")
        if type(rope.get("mrope_interleaved", True)) is not bool:
            raise ValueError("mRoPE interleaving flag must be boolean")
        inv = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
        )
        if not bool(torch.isfinite(inv).all()):
            raise ValueError("Inverse frequencies must be finite FP32")
        return cls(dim, maximum, inv, tuple(section) if section is not None else None)

    def tables(self, backend, *, position, tokens, dtype, device):
        if type(position) is not int or position < 0:
            raise ValueError("Position must be a nonnegative integer")
        _shape(tokens, 1)
        if position + tokens > self.maximum_position:
            raise ValueError("Positions exceed configured context")
        target = backend.validate_device(dtype, device)
        _outside_capture(target)
        # Match the original CPU FP32 inverse-frequency initialization, then
        # compute phases/trig on the execution device with autocast disabled.
        inv = self.inverse_frequency.to(target)
        with torch.autocast(device_type=target.type, enabled=False):
            phase = (
                torch.arange(
                    position, position + tokens, dtype=torch.float32, device=target
                )[:, None]
                * inv[None, :]
            )
            cos, sin = (
                f(phase).to(dtype).float().contiguous() for f in (torch.cos, torch.sin)
            )
        if target.type == "cuda":
            torch.cuda.current_stream(target).synchronize()
        return cos, sin
