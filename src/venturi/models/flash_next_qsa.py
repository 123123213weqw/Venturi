"""Bounded floating QSA indexer reference; no native sparse-attention executor.

Pool blocks of VISIBLE token indices, not fixed absolute-position blocks. Raw
unrotated indexer keys are explicit caller state; this module never commits a
request cache. Dense masks are only a small-case correctness representation.
"""

from dataclasses import dataclass
import math

from safetensors import safe_open
import torch
from torch import nn

from ..dtypes import torch_dtype
from ..flash_next import _positive
from .flash_next_residual import GroupedZeroCenteredRMSNorm


@dataclass(frozen=True)
class QSAIndexerGeometry:
    hidden_size: int
    query_heads: int
    head_dim: int
    token_budget: int
    compress_ratio: int
    eps: float = 1e-6

    def __post_init__(self):
        for key in ("hidden_size", "query_heads", "head_dim", "token_budget", "compress_ratio"):
            _positive(getattr(self, key), key)
        if self.token_budget % self.compress_ratio or self.head_dim % 2:
            raise ValueError("QSA budget must divide into blocks, and head dimension must be even")
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Positive finite QSA RMS epsilon required")

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("model_type") != "qwen4_exp":
            raise ValueError("Explicit qwen4_exp configuration required")
        c = raw.get("text_config")
        if not isinstance(c, dict) or c.get("model_type") != "qwen4_exp_text":
            raise ValueError("Explicit qwen4_exp_text configuration required")
        if c.get("indexer_kv_heads") != 1:
            raise ValueError("Only the qualified single-KV-head QSA indexer is supported")
        return cls(c.get("hidden_size"), c.get("indexer_n_heads"), c.get("indexer_head_dim"),
                   c.get("indexer_budget"), c.get("indexer_compress_ratio"), c.get("rms_norm_eps", 1e-6))


def partial_rope(x, cos, sin, *, unsqueeze_dim):
    """Use the supplied rotary width, NOT indexer_dim * partial_rotary_factor.

    The official main head is256/rotary64 but the indexer head is128 and uses
    that same64-wide position embedding. Recomputing128*0.25 would be wrong.
    """
    width = cos.shape[-1]
    if sin.shape != cos.shape or width <= 0 or width % 2 or width > x.shape[-1]:
        raise ValueError("Invalid QSA partial rotary geometry")
    if any(t.dtype != x.dtype or t.device != x.device for t in (cos, sin)):
        raise ValueError("QSA rotary tensors must share activation dtype/device")
    rotated, untouched = x[..., :width], x[..., width:]
    half = width // 2
    alternate = torch.cat((-rotated[..., half:], rotated[..., :half]), -1)
    return torch.cat((rotated * cos.unsqueeze(unsqueeze_dim) + alternate * sin.unsqueeze(unsqueeze_dim), untouched), -1)


class FlashNextQSAIndexer(nn.Module):
    """Explicit projection/selection mask and proposed raw-key state.

    Returns (selected_mask, proposed_raw_keys). Caller must own request-private
    state and commit/truncate only accepted tokens; no implicit update occurs.
    No main attention Q/K/V, final gate, KV paging or full-model forward here.
    """

    def __init__(self, geometry, *, max_mask_elements=4 * 2**20):
        super().__init__()
        if type(max_mask_elements) is not int or not 0 < max_mask_elements <= 64 * 2**20:
            raise ValueError("Explicit bounded reference-mask budget required")
        self.geometry, self.max_mask_elements = geometry, max_mask_elements
        self.index_qk_proj = nn.Linear(geometry.hidden_size, (geometry.query_heads + 1) * geometry.head_dim, bias=False)
        self.q_layernorm = GroupedZeroCenteredRMSNorm(geometry.head_dim, 1, geometry.eps)
        self.k_layernorm = GroupedZeroCenteredRMSNorm(geometry.head_dim, 1, geometry.eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, *, raw_past=None):
        g = self.geometry
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != g.hidden_size or min(hidden_states.shape[:2]) < 1:
            raise ValueError("QSA requires nonempty batch/sequence/hidden input")
        batch, length, _ = hidden_states.shape
        past_length = 0 if raw_past is None else raw_past.shape[1] if raw_past.ndim == 3 else -1
        kv_length = past_length + length
        if attention_mask.shape != (batch, 1, length, kv_length) or attention_mask.dtype not in (
            torch.bool, torch.float16, torch.bfloat16, torch.float32
        ) or attention_mask.device != hidden_states.device:
            raise ValueError("Explicit4D visibility/additive mask must cover current and past indexer keys")
        if attention_mask.numel() > self.max_mask_elements or (
            batch * length * (g.token_budget + g.compress_ratio - 1) > self.max_mask_elements
        ):
            raise ValueError("QSA reference exceeds explicit dense-mask/selection budget")
        if hidden_states.dtype != self.index_qk_proj.weight.dtype or hidden_states.device != self.index_qk_proj.weight.device:
            raise ValueError("QSA input and projection weights must share dtype/device")
        if raw_past is not None and (raw_past.shape != (batch, past_length, g.head_dim) or
            raw_past.dtype != hidden_states.dtype or raw_past.device != hidden_states.device):
            raise ValueError("QSA raw past geometry/dtype/device mismatch")
        cos, sin = position_embeddings
        if cos.ndim != 3 or cos.shape[:2] != (batch, kv_length) or sin.shape != cos.shape:
            raise ValueError("Full current+past position embeddings required")
        qk = self.index_qk_proj(hidden_states)
        query, keys = torch.split(qk, [g.query_heads * g.head_dim, g.head_dim], -1)
        query = self.q_layernorm(query.reshape(batch, length, g.query_heads, g.head_dim))
        query = partial_rope(query, cos[:, -length:], sin[:, -length:], unsqueeze_dim=2)
        # A split view would retain the entire Q+K projection allocation in a
        # request's persistent state. Own only the compact raw-key allocation.
        raw_keys = keys.clone() if raw_past is None else torch.cat((raw_past, keys), 1)
        visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        selected = torch.full((batch, length, g.token_budget + g.compress_ratio - 1), -1,
                              dtype=torch.int32, device=hidden_states.device)
        for b in range(batch):
            for t in range(length):
                local = torch.nonzero(visible[b, 0, t], as_tuple=False).flatten()
                blocks = local.numel() // g.compress_ratio
                if blocks:
                    indices = local[:blocks * g.compress_ratio].view(blocks, g.compress_ratio)
                    key_groups = raw_keys[b].index_select(0, indices.flatten()).view(blocks, g.compress_ratio, g.head_dim)
                    pooled = self.k_layernorm(key_groups.float().mean(1).to(raw_keys.dtype))
                    starts = indices[:, 0]
                    block_keys = partial_rope(pooled.unsqueeze(1), cos[b].index_select(0, starts),
                                              sin[b].index_select(0, starts), unsqueeze_dim=1).squeeze(1)
                    scores = torch.matmul(query[b, t].float(), block_keys.float().T).T
                    scores = torch.relu(scores).sum(-1) / math.sqrt(g.head_dim)
                    keep = scores.topk(min(g.token_budget // g.compress_ratio, blocks), dim=0).indices
                    chosen = indices.index_select(0, keep).flatten()
                else:
                    chosen = local[:0]
                chosen = torch.cat((chosen, local[blocks * g.compress_ratio:])).to(torch.int32)
                selected[b, t, :chosen.numel()] = chosen
        mask = torch.zeros((batch, length, kv_length + 1), device=attention_mask.device, dtype=torch.bool)
        scatter = torch.where(selected >= 0, selected, kv_length)
        # Torch2.5 on V100 requires int64 scatter indices; retain compact int32
        # selection storage and widen only this bounded scatter input.
        mask = mask.scatter(-1, scatter.to(torch.int64), True)[..., :kv_length].unsqueeze(1)
        if attention_mask.is_floating_point():
            mask = torch.where(mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return mask, raw_keys

    @classmethod
    def from_checkpoint(cls, checkpoint, geometry, prefix, *, max_weight_bytes=32 * 2**20,
                        max_mask_elements=4 * 2**20):
        g = geometry
        specs = {}
        shapes = {"index_qk_proj.weight": ((g.query_heads + 1) * g.head_dim, g.hidden_size),
                  "q_layernorm.weight": (g.head_dim,), "k_layernorm.weight": (g.head_dim,)}
        for key, shape in shapes.items():
            spec = checkpoint.tensors.get(prefix + "." + key)
            if spec is None or spec.shape != shape or spec.dtype not in ("BF16", "F16", "F32"):
                raise ValueError("Original floating QSA indexer weight required")
            specs[key] = spec
        if len({s.dtype for s in specs.values()}) != 1 or type(max_weight_bytes) is not int or (
            max_weight_bytes <= 0 or sum(s.nbytes for s in specs.values()) > max_weight_bytes
        ):
            raise ValueError("QSA weights must share dtype and fit the explicit budget")
        result = cls(g, max_mask_elements=max_mask_elements).to(dtype=torch_dtype(next(iter(specs.values())).dtype))
        with torch.no_grad():
            parameters = dict(result.named_parameters())
            for key, spec in specs.items():
                with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(spec.name)
                    if tuple(value.shape) != spec.shape or value.dtype != parameters[key].dtype:
                        raise ValueError("QSA backing metadata changed")
                    parameters[key].copy_(value)
        return result.eval()
