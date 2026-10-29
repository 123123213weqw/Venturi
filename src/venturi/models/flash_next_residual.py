"""Floating Qwen4-Exp gated-residual reference, not a native model forward.

Checkpoint RMS weights are zero-centered (1+w); RMS groups are independent
residual streams, not one normalization over the concatenated HC dimension.
Preserve explicit BF16 rounding boundaries before attempting fused kernels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GroupedZeroCenteredRMSNorm(nn.Module):
    def __init__(self, hidden_size, streams, eps=1e-6):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (hidden_size, streams)):
            raise ValueError("Positive hidden size and stream count required")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Positive finite RMS epsilon required")
        self.hidden_size, self.streams, self.eps = hidden_size, streams, eps
        self.weight = nn.Parameter(torch.zeros(hidden_size * streams))

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.hidden_size * self.streams:
            raise ValueError("Expected flattened residual streams")
        if x.dtype != self.weight.dtype or x.device != self.weight.device or not x.is_floating_point():
            raise ValueError("Residual and weights must have the same floating dtype/device")
        grouped = x.float().reshape(*x.shape[:-1], self.streams, self.hidden_size)
        normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)
        return (normalized.flatten(-2) * (1.0 + self.weight.float())).to(x.dtype)


class FlashNextGatedResidual(nn.Module):
    """Read/mix four-stream residuals and optionally produce injection gates.

    Input/output prefix dimensions are arbitrary (e.g. batch, sequence). The
    final mixer has no injection weights. No request/KV state is stored here.
    Returned residual aliases the input deliberately; injection is out-of-place.
    """

    def __init__(self, hidden_size, streams, lowrank, *, eps=1e-6, use_combine=True):
        super().__init__()
        if type(lowrank) is not int or lowrank < 1 or type(use_combine) is not bool:
            raise ValueError("Positive lowrank and explicit combine flag required")
        self.hidden_size, self.streams = hidden_size, streams
        self.hc_norm = GroupedZeroCenteredRMSNorm(hidden_size, streams, eps)
        total = hidden_size * streams
        self.input_mix_weight_down = nn.Linear(total, lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(lowrank, total, bias=False)
        self.block_inject_weight = nn.Linear(total, streams, bias=False) if use_combine else None

    def forward(self, hyper_input):
        normalized = self.hc_norm(hyper_input)
        low = F.silu(self.input_mix_weight_down(normalized) / self.streams)
        gates = torch.sigmoid(self.input_mix_weight_up(low)).unflatten(-1, (self.streams, self.hidden_size))
        mixed = (gates * normalized.unflatten(-1, (self.streams, self.hidden_size))).mean(-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * torch.sigmoid(self.block_inject_weight(normalized) / self.streams)
        return mixed, hyper_input, injection

    def inject(self, block_output, hyper_input, injection_weights):
        if self.block_inject_weight is None:
            raise ValueError("Final residual mixer does not inject a block output")
        prefix = hyper_input.shape[:-1]
        if hyper_input.shape[-1:] != (self.streams * self.hidden_size,) or (
            block_output.shape != (*prefix, self.hidden_size) or
            injection_weights.shape != (*prefix, self.streams)
        ):
            raise ValueError("Block output/residual/injection geometry mismatch")
        if any(t.dtype != hyper_input.dtype or t.device != hyper_input.device
               for t in (block_output, injection_weights)):
            raise ValueError("Injection tensors must share dtype/device")
        injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        return hyper_input + injection.flatten(-2)

    @classmethod
    def from_checkpoint(cls, checkpoint, config, prefix, *, eps, use_combine=True):
        """Load only the small original floating GR tensors, never expert/PLE data."""
        from safetensors import safe_open
        from ..dtypes import torch_dtype

        total, low = config.hc_count * config.hidden_size, config.hc_lowrank
        shapes = {"hc_norm.weight": (total,), "input_mix_weight_down.weight": (low, total),
                  "input_mix_weight_up.weight": (total, low)}
        if use_combine:
            shapes["block_inject_weight.weight"] = (config.hc_count, total)
        specs = {}
        for key, shape in shapes.items():
            spec = checkpoint.tensors.get(f"{prefix}.{key}")
            if spec is None or spec.shape != shape or spec.dtype not in ("F32", "F16", "BF16"):
                raise ValueError(f"Original floating gated-residual tensor required: {key}")
            specs[key] = spec
        if len({s.dtype for s in specs.values()}) != 1:
            raise ValueError("Gated-residual weights must share a dtype")
        result = cls(config.hidden_size, config.hc_count, config.hc_lowrank,
                     eps=eps, use_combine=use_combine).to(dtype=torch_dtype(next(iter(specs.values())).dtype))
        # Own bounded weight allocations after closing the source mappings.
        with torch.no_grad():
            parameters = dict(result.named_parameters())
            for key, spec in specs.items():
                with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(spec.name)
                    if tuple(value.shape) != spec.shape or value.dtype != parameters[key].dtype:
                        raise ValueError("Gated-residual backing metadata changed")
                    parameters[key].copy_(value)
        return result.eval()
