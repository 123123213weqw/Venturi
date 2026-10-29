"""Floating PLE conditioning/short conv on externally gathered CPU-table rows.

Does not own/materialize the huge ngram table or commit token history. PLE
injects before attention HC; its dilated conv state is separate from GDN state.
"""

from dataclasses import dataclass
import math

from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F

from ..dtypes import torch_dtype
from ..flash_next import _positive
from .flash_next_residual import GroupedZeroCenteredRMSNorm


@dataclass(frozen=True)
class PLEGeometry:
    hidden_size: int
    streams: int
    embedding_dim: int
    conv_kernel: int
    dilation: int
    eps: float = 1e-6

    def __post_init__(self):
        for name in ("hidden_size", "streams", "embedding_dim", "conv_kernel", "dilation"):
            _positive(getattr(self, name), name)
        if not 2 <= self.conv_kernel <= 16 or self.dilation > 16:
            raise ValueError("Bounded nontrivial dilated PLE convolution required")
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Positive finite PLE epsilon required")

    @property
    def state_length(self):
        return (self.conv_kernel - 1) * self.dilation

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("model_type") != "qwen4_exp":
            raise ValueError("Explicit qwen4_exp configuration required")
        c = raw.get("text_config")
        if not isinstance(c, dict) or c.get("model_type") != "qwen4_exp_text":
            raise ValueError("Explicit qwen4_exp_text configuration required")
        return cls(*(c.get(k) for k in ("hidden_size", "hc_count", "ple_embed_dim", "ple_conv_kernel_size", "ngram_size")),
                   eps=c.get("rms_norm_eps", 1e-6))


@dataclass(frozen=True)
class PLEState:
    conv: torch.Tensor  # [B, HC*H, (kernel-1)*dilation], normalized gated values
    tokens_seen: int

    def snapshot(self):
        return PLEState(self.conv.clone(), self.tokens_seen)


def gather_ple_rows(table, proposals, *, device, dtype):
    """Coalesce rows across equally sized private proposals, explicit H2D/cast.

    No automatic pinning, prefetch, selector or commit. Reader enforces its
    global token/read bounds. This is an integration bridge, not a speed claim.
    """
    proposals = tuple(proposals)
    if not proposals or not proposals[0].token_ids or len({len(p.token_ids) for p in proposals}) != 1:
        raise ValueError("PLE reference batch needs nonempty equal-length proposals")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Explicit floating PLE execution dtype required")
    return torch.stack(table.lookup_many(proposals)).to(device=device, dtype=dtype)


class FlashNextPLE(nn.Module):
    """External rows -> stream-specific signed-sqrt gates -> dilated short conv.

    Returns conditioning output and private proposed short-conv state. Masking
    current inputs does not skip history advancement or past-conv contributions.
    Caller owns separate NGramRequest history and accepted-prefix replay/commit.
    """

    def __init__(self, geometry, *, max_work_elements=16 * 2**20, max_state_bytes=64 * 2**20):
        super().__init__()
        if type(max_work_elements) is not int or not 0 < max_work_elements <= 64 * 2**20 or (
            type(max_state_bytes) is not int or not 0 < max_state_bytes <= 2**30
        ):
            raise ValueError("Explicit bounded PLE work/state budget required")
        self.geometry, self.max_work_elements, self.max_state_bytes = geometry, max_work_elements, max_state_bytes
        g = geometry
        total = g.hidden_size * g.streams
        self.key_proj = nn.Linear(g.embedding_dim, total, bias=False)
        self.value_proj = nn.Linear(g.embedding_dim, g.hidden_size, bias=False)
        self.norm_key = GroupedZeroCenteredRMSNorm(g.hidden_size, g.streams, g.eps)
        self.norm_query = GroupedZeroCenteredRMSNorm(g.hidden_size, g.streams, g.eps)
        self.norm_conv = GroupedZeroCenteredRMSNorm(g.hidden_size, g.streams, g.eps)
        self.conv1d = nn.Conv1d(total, total, g.conv_kernel, groups=total, dilation=g.dilation, bias=False)

    def forward(self, hidden_states, embeddings, *, past=None, conv_mask=None):
        g = self.geometry
        if self.training:
            raise ValueError("PLE reference is inference-only; call eval()")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != g.streams * g.hidden_size or min(hidden_states.shape[:2]) < 1:
            raise ValueError("Nonempty flattened-HC PLE input required")
        batch, length, _ = hidden_states.shape
        if embeddings.shape != (batch, length, g.embedding_dim) or any(
            t.dtype != self.key_proj.weight.dtype or t.device != self.key_proj.weight.device for t in (hidden_states, embeddings)
        ):
            raise ValueError("PLE input/embedding/weight geometry, dtype or device mismatch")
        if conv_mask is not None and (conv_mask.shape != (batch, length) or conv_mask.dtype != torch.bool or
                                      conv_mask.device != hidden_states.device):
            raise ValueError("PLE conv mask must be2D bool on the input device")
        if past is not None and (not isinstance(past, PLEState) or type(past.tokens_seen) is not int or past.tokens_seen < 0):
            raise ValueError("Explicit PLEState with nonnegative tokens_seen required")
        if past is not None and (past.conv.shape != (batch, g.streams*g.hidden_size, g.state_length) or
            past.conv.dtype != hidden_states.dtype or past.conv.device != hidden_states.device):
            raise ValueError("PLE short-conv past geometry/dtype/device mismatch")
        state_elements = batch * g.streams * g.hidden_size * g.state_length
        work_elements = state_elements * 2 + batch * length * (g.embedding_dim + g.streams*g.hidden_size*10)
        if state_elements * hidden_states.element_size() > self.max_state_bytes or work_elements > self.max_work_elements:
            raise ValueError("PLE exceeds explicit work/state budget")
        key = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (g.streams, g.hidden_size))
        value = self.value_proj(embeddings)
        query = self.norm_query(hidden_states).unflatten(-1, (g.streams, g.hidden_size))
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(g.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        normalized = self.norm_conv(gated_value)
        if conv_mask is not None:
            gated_value = (gated_value * conv_mask[:, :, None]).to(gated_value.dtype)
            normalized = (normalized * conv_mask[:, :, None]).to(normalized.dtype)
        current = normalized.transpose(1, 2)
        context = current if past is None else torch.cat((past.conv, current), -1)
        if past is None and length < g.state_length:
            context = F.pad(context, (g.state_length - length, 0))
        proposed = context[..., -g.state_length:].contiguous().clone()
        conv_input = F.pad(context, (g.state_length, 0))[..., -(g.state_length + length):]
        convolved = F.silu(self.conv1d(conv_input)).transpose(1, 2)
        output = gated_value + convolved
        return output, PLEState(proposed, (0 if past is None else past.tokens_seen) + length)

    @classmethod
    def from_checkpoint(cls, checkpoint, geometry, prefix, *, max_weight_bytes=128 * 2**20, **budgets):
        with torch.device("meta"):
            prototype = cls(geometry, **budgets)
        specs = {}
        for name, parameter in prototype.named_parameters():
            spec = checkpoint.tensors.get(prefix + "." + name)
            if spec is None or spec.shape != tuple(parameter.shape) or spec.dtype not in ("BF16", "F16", "F32"):
                raise ValueError("Original floating PLE conditioning tensor required")
            specs[name] = spec
        if len({s.dtype for s in specs.values()}) != 1 or type(max_weight_bytes) is not int or (
            max_weight_bytes <= 0 or sum(s.nbytes for s in specs.values()) > max_weight_bytes
        ):
            raise ValueError("PLE small weights must share dtype and fit explicit byte budget")
        dtype = torch_dtype(next(iter(specs.values())).dtype)
        result = prototype.to_empty(device="cpu").to(dtype=dtype)
        with torch.no_grad():
            for name, parameter in result.named_parameters():
                spec = specs[name]
                with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(spec.name)
                    if tuple(value.shape) != spec.shape or value.dtype != dtype:
                        raise ValueError("PLE backing metadata changed")
                    parameter.copy_(value)
        return result.eval()
