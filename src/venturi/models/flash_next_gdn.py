"""Floating Flash-Next GDN block with explicit private convolution/RNN state.

Original SiLU causal conv, K->V head expansion and sigmoid output gate. Not
the dense qwen3_5 native frontend and not a native/full-model implementation.
Padding follows source multiplication semantics, NOT an inactive-slot skip.
"""

from dataclasses import dataclass
import math

from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F

from ..dtypes import torch_dtype
from ..flash_next import _positive
from .flash_next_delta import SigmoidGatedRMSNorm, gated_delta_rule


@dataclass(frozen=True)
class GDNGeometry:
    hidden_size: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    eps: float = 1e-6

    def __post_init__(self):
        for name in ("hidden_size", "key_heads", "value_heads", "key_dim", "value_dim", "conv_kernel"):
            _positive(getattr(self, name), name)
        if self.value_heads % self.key_heads or self.conv_kernel > 16:
            raise ValueError("GDN needs integral K->V head expansion and a bounded conv kernel")
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Positive finite GDN epsilon required")

    @property
    def conv_channels(self):
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("model_type") != "qwen4_exp":
            raise ValueError("Explicit qwen4_exp configuration required")
        c = raw.get("text_config")
        if not isinstance(c, dict) or c.get("model_type") != "qwen4_exp_text":
            raise ValueError("Explicit qwen4_exp_text configuration required")
        if c.get("hidden_act") != "silu" or c.get("output_gate_type") != "sigmoid":
            raise ValueError("Only qualified SiLU-conv/sigmoid-output GDN is supported")
        return cls(*(c.get(name) for name in ("hidden_size", "linear_num_key_heads", "linear_num_value_heads",
                   "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim")),
                   eps=c.get("rms_norm_eps", 1e-6))


@dataclass(frozen=True)
class GDNState:
    conv: torch.Tensor  # [B, 2*HK*DK+HV*DV, kernel], raw projection history
    recurrent: torch.Tensor  # FP32 [B, HV, DK, DV]
    tokens_seen: int

    def snapshot(self):
        """Owned base for proposal rollback/replay; no token-index matrix crop."""
        return GDNState(self.conv.clone(), self.recurrent.clone(), self.tokens_seen)


class FlashNextGDN(nn.Module):
    """Inference reference block returning (output, proposed_state), no commit.

Follows pinned chunk64 prefill/cached-multi-token and recurrent cached-single
mode selection. No record-past/export fallback, ragged scheduler or native op.
To reject a suffix restore a base snapshot and replay accepted tokens; slicing
the final recurrent matrix is not a valid state rollback.
"""

    def __init__(self, geometry, *, max_work_elements=64 * 2**20, max_state_bytes=128 * 2**20):
        super().__init__()
        if type(max_work_elements) is not int or not 0 < max_work_elements <= 64 * 2**20 or (
            type(max_state_bytes) is not int or not 0 < max_state_bytes <= 2**30
        ):
            raise ValueError("Explicit bounded GDN work/state budget required")
        self.geometry, self.max_work_elements, self.max_state_bytes = geometry, max_work_elements, max_state_bytes
        g = geometry
        self.conv1d = nn.Conv1d(g.conv_channels, g.conv_channels, g.conv_kernel, groups=g.conv_channels,
                                padding=g.conv_kernel - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(g.value_heads))
        self.A_log = nn.Parameter(torch.zeros(g.value_heads))
        self.norm = SigmoidGatedRMSNorm(g.value_dim, g.eps)
        self.out_proj = nn.Linear(g.value_heads * g.value_dim, g.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(g.hidden_size, g.conv_channels, bias=False)
        self.in_proj_z = nn.Linear(g.hidden_size, g.value_heads * g.value_dim, bias=False)
        self.in_proj_b = nn.Linear(g.hidden_size, g.value_heads, bias=False)
        self.in_proj_a = nn.Linear(g.hidden_size, g.value_heads, bias=False)

    def forward(self, hidden_states, *, past=None, attention_mask=None):
        g = self.geometry
        if self.training:
            raise ValueError("GDN reference is inference-only; call eval()")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != g.hidden_size or min(hidden_states.shape[:2]) < 1:
            raise ValueError("GDN requires nonempty batch/sequence/hidden input")
        if hidden_states.dtype != self.in_proj_qkv.weight.dtype or hidden_states.device != self.in_proj_qkv.weight.device:
            raise ValueError("GDN input and weights must share dtype/device")
        batch, length, _ = hidden_states.shape
        if attention_mask is not None and (attention_mask.shape != (batch, length) or
            attention_mask.dtype != torch.bool or attention_mask.device != hidden_states.device):
            raise ValueError("GDN source padding mask must be2D bool on the input device")
        if past is not None:
            if not isinstance(past, GDNState) or type(past.tokens_seen) is not int or past.tokens_seen < 0:
                raise ValueError("Explicit GDNState with nonnegative tokens_seen required")
            if past.conv.shape != (batch, g.conv_channels, g.conv_kernel) or (
                past.recurrent.shape != (batch, g.value_heads, g.key_dim, g.value_dim)
            ) or past.conv.dtype != hidden_states.dtype or past.recurrent.dtype != torch.float32 or any(
                t.device != hidden_states.device for t in (past.conv, past.recurrent)
            ):
                raise ValueError("GDN conv/recurrent state geometry/dtype/device mismatch")
        mode = "recurrent" if past is not None and length == 1 else "chunk"
        padded = length if mode == "recurrent" else ((length + 63) // 64) * 64
        state_bytes = batch * (g.conv_channels * g.conv_kernel * hidden_states.element_size() +
                               g.value_heads * g.key_dim * g.value_dim * 4)
        core = batch * g.value_heads * (g.key_dim * g.value_dim + padded * (4*g.key_dim + 4*g.value_dim + 4))
        if mode == "chunk":
            core += batch * g.value_heads * padded * 64 * 4
        front = batch * ((length + g.conv_kernel) * g.conv_channels +
                         length * (g.conv_channels + g.value_heads * g.value_dim + 2 * g.value_heads))
        if state_bytes > self.max_state_bytes or core + front > self.max_work_elements:
            raise ValueError("GDN reference exceeds explicit state/work budget")
        # Source semantics: mask hidden input, but still process/decay state.
        # Inactive serving slots must not be padded/committed through this API.
        x = hidden_states if attention_mask is None else (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        raw = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(batch, length, g.value_heads, g.value_dim)
        b, a = self.in_proj_b(x), self.in_proj_a(x)
        context = raw if past is None else torch.cat((past.conv, raw), -1)
        if past is None and length < g.conv_kernel:
            context = F.pad(context, (g.conv_kernel - length, 0))
        proposed_conv = context[..., -g.conv_kernel:].contiguous().clone()
        if mode == "recurrent":
            # Match causal_conv1d_update: slice BEFORE activation. Activating
            # the extra convolution output then slicing changes CPU SIMD SiLU
            # rounding (observed with actual B8 cached weights), not just work.
            convolved = F.conv1d(context.to(self.conv1d.weight.dtype), self.conv1d.weight,
                                groups=g.conv_channels)[..., -length:]
        else:
            convolved = F.conv1d(context.to(self.conv1d.weight.dtype), self.conv1d.weight,
                                padding=g.conv_kernel - 1, groups=g.conv_channels)[..., :context.shape[-1]]
        convolved = F.silu(convolved)[..., -length:].to(hidden_states.dtype).transpose(1, 2)
        q, k, v = convolved.split((g.key_heads * g.key_dim, g.key_heads * g.key_dim, g.value_heads * g.value_dim), -1)
        q = q.reshape(batch, length, g.key_heads, g.key_dim)
        k = k.reshape(batch, length, g.key_heads, g.key_dim)
        v = v.reshape(batch, length, g.value_heads, g.value_dim)
        beta = b.sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        expansion = g.value_heads // g.key_heads
        if expansion > 1:
            q, k = q.repeat_interleave(expansion, 2), k.repeat_interleave(expansion, 2)
        core_output, proposed_recurrent = gated_delta_rule(q, k, v, decay, beta,
            initial_state=None if past is None else past.recurrent, mode=mode, normalize_qk=True,
            max_work_elements=self.max_work_elements)
        gated = self.norm(core_output.reshape(-1, g.value_dim), z.reshape(-1, g.value_dim))
        output = self.out_proj(gated.reshape(batch, length, -1))
        return output, GDNState(proposed_conv, proposed_recurrent, (0 if past is None else past.tokens_seen) + length)

    @classmethod
    def from_checkpoint(cls, checkpoint, geometry, prefix, *, max_weight_bytes=128 * 2**20, **budgets):
        with torch.device("meta"):
            prototype = cls(geometry, **budgets)
        specs = {}
        for name, parameter in prototype.named_parameters():
            spec = checkpoint.tensors.get(prefix + "." + name)
            if spec is None or spec.shape != tuple(parameter.shape) or spec.dtype not in ("BF16", "F16", "F32"):
                raise ValueError("Original floating Flash-Next GDN tensor required")
            specs[name] = spec
        if len({s.dtype for s in specs.values()}) != 1 or type(max_weight_bytes) is not int or (
            max_weight_bytes <= 0 or sum(s.nbytes for s in specs.values()) > max_weight_bytes
        ):
            raise ValueError("GDN weights must share dtype and fit explicit byte budget")
        dtype = torch_dtype(next(iter(specs.values())).dtype)
        result = prototype.to_empty(device="cpu").to(dtype=dtype)
        with torch.no_grad():
            for name, parameter in result.named_parameters():
                spec = specs[name]
                with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(spec.name)
                    if tuple(value.shape) != spec.shape or value.dtype != dtype:
                        raise ValueError("GDN backing metadata changed")
                    parameter.copy_(value)
        return result.eval()
