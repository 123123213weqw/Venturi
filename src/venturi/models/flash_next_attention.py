"""Bounded floating QSA attention reference, not a native sparse executor.

Indexer raw keys and main normalized/rotated K/V remain separate. Proposed
state never mutates past and must be committed by the caller, not this module.
Dense attention/masks are correctness representations, not a memory-efficient
262k-context implementation. Eager is the unchanged default; SDPA is explicit.
"""

from dataclasses import dataclass

from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F

from ..dtypes import torch_dtype
from ..flash_next import _positive
from .flash_next_qsa import FlashNextQSAIndexer, QSAIndexerGeometry, partial_rope
from .flash_next_residual import GroupedZeroCenteredRMSNorm


@dataclass(frozen=True)
class QSAAttentionGeometry:
    indexer: QSAIndexerGeometry
    query_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self):
        if not isinstance(self.indexer, QSAIndexerGeometry):
            raise ValueError("Explicit qualified QSA indexer geometry required")
        for key in ("query_heads", "kv_heads", "head_dim"):
            _positive(getattr(self, key), key)
        if self.query_heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("QSA query heads must divide into KV groups; even head dimension required")

    @classmethod
    def from_dict(cls, raw):
        indexer = QSAIndexerGeometry.from_dict(raw)
        c = raw["text_config"]
        if c.get("attention_bias", False) is not False:
            raise ValueError("Only bias-free original QSA attention is qualified")
        return cls(indexer, c.get("num_attention_heads"), c.get("num_key_value_heads"), c.get("head_dim"))


@dataclass(frozen=True)
class QSAState:
    raw_indexer_keys: torch.Tensor  # [B, T, indexer_dim], before norm/rotation
    keys: torch.Tensor  # [B, KV_heads, T, D], after norm/rotation
    values: torch.Tensor  # [B, KV_heads, T, D]

    def truncate(self, length):
        """Own an accepted absolute prefix; rejected suffixes keep no aliases."""
        if type(length) is not int or not 0 <= length <= self.raw_indexer_keys.shape[1]:
            raise ValueError("Accepted QSA state prefix is out of bounds")
        return QSAState(self.raw_indexer_keys[:, :length].clone(),
                        self.keys[:, :, :length].clone(), self.values[:, :, :length].clone())


class FlashNextQSAAttention(nn.Module):
    """Q/K/V, grouped-query eager attention, per-head gate and output projection.

Returns (output, proposed_state, probabilities). All dense scratch/state have
explicit bounds. Floating additive mask semantics match pinned eager upstream,
    including NaNs for some fully masked rows after finite-min masks add to -inf.
    Explicit SDPA returns no probabilities and has separately checked empty-row
    semantics; it is not a silent replacement/repair of eager or a serving route.
"""

    def __init__(self, geometry, *, max_score_elements=4 * 2**20, max_state_bytes=128 * 2**20,
                 max_expanded_kv_bytes=512 * 2**20, attention_backend="eager"):
        super().__init__()
        if attention_backend not in ("eager", "sdpa"):
            raise ValueError("Explicit eager/sdpa attention reference backend required")
        self.attention_backend = attention_backend
        for name, value, limit in (("score", max_score_elements, 64 * 2**20),
                                   ("state", max_state_bytes, 2**30),
                                   ("expanded-KV", max_expanded_kv_bytes, 2**30)):
            if type(value) is not int or not 0 < value <= limit:
                raise ValueError(f"Explicit bounded QSA {name} budget required")
        self.geometry = geometry
        self.max_score_elements, self.max_state_bytes = max_score_elements, max_state_bytes
        self.max_expanded_kv_bytes = max_expanded_kv_bytes
        g, h = geometry, geometry.indexer.hidden_size
        self.indexer = FlashNextQSAIndexer(g.indexer, max_mask_elements=max_score_elements)
        self.q_proj = nn.Linear(h, g.query_heads * g.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(h, g.kv_heads * g.head_dim, bias=False)
        self.v_proj = nn.Linear(h, g.kv_heads * g.head_dim, bias=False)
        self.o_proj = nn.Linear(g.query_heads * g.head_dim, h, bias=False)
        self.q_norm = GroupedZeroCenteredRMSNorm(g.head_dim, 1, g.indexer.eps)
        self.k_norm = GroupedZeroCenteredRMSNorm(g.head_dim, 1, g.indexer.eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, *, past=None):
        g = self.geometry
        if self.training:
            raise ValueError("QSA reference is inference-only; call eval()")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != g.indexer.hidden_size or min(hidden_states.shape[:2]) < 1:
            raise ValueError("Nonempty QSA batch/sequence/hidden input required")
        batch, length, _ = hidden_states.shape
        if hidden_states.dtype != self.q_proj.weight.dtype or hidden_states.device != self.q_proj.weight.device:
            raise ValueError("QSA input and weights must share dtype/device")
        if past is not None and not isinstance(past, QSAState):
            raise ValueError("Explicit QSAState required")
        past_length = 0
        if past is not None:
            if past.raw_indexer_keys.ndim != 3:
                raise ValueError("Invalid QSA past shape")
            past_length = past.raw_indexer_keys.shape[1]
            shapes = ((batch, past_length, g.indexer.head_dim),
                      (batch, g.kv_heads, past_length, g.head_dim),
                      (batch, g.kv_heads, past_length, g.head_dim))
            if any(t.shape != s or t.dtype != hidden_states.dtype or t.device != hidden_states.device
                   for t, s in zip((past.raw_indexer_keys, past.keys, past.values), shapes)):
                raise ValueError("QSA past geometry/dtype/device mismatch")
        kv_length = past_length + length
        if batch * g.query_heads * length * kv_length > self.max_score_elements or (
            batch * kv_length * (g.indexer.head_dim + 2 * g.kv_heads * g.head_dim) * hidden_states.element_size()
            > self.max_state_bytes
        ):
            raise ValueError("QSA attention exceeds explicit score/state budget")
        # Eager GQA materializes repeated K/V unlike a native grouped kernel.
        # State/score bounds alone do not bound this extra head-expanded memory.
        if 2 * batch * g.query_heads * kv_length * g.head_dim * hidden_states.element_size() > self.max_expanded_kv_bytes:
            raise ValueError("QSA eager attention exceeds explicit expanded-KV budget")
        mask_dtype_ok = attention_mask.dtype == hidden_states.dtype and attention_mask.is_floating_point()
        if self.attention_backend == "sdpa" and attention_mask.dtype == torch.bool:
            mask_dtype_ok = True
        if attention_mask.shape != (batch, 1, length, kv_length) or not mask_dtype_ok or attention_mask.device != hidden_states.device:
            raise ValueError("QSA requires explicit4D additive mask in activation dtype/device (bool also allowed for explicit SDPA)")
        # Indexer also validates full position geometry/dtype/device before projections.
        selection, raw = self.indexer(hidden_states, position_embeddings, attention_mask,
                                      raw_past=None if past is None else past.raw_indexer_keys)
        combined_mask = attention_mask + selection if attention_mask.is_floating_point() else attention_mask & selection
        # Critical: interleaved [head, query_D, gate_D], NOT two global halves.
        query, gate = self.q_proj(hidden_states).view(batch, length, g.query_heads, 2 * g.head_dim).chunk(2, -1)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, length, g.kv_heads, g.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, g.kv_heads, g.head_dim).transpose(1, 2)
        cos, sin = (v[:, -length:] for v in position_embeddings)
        query = partial_rope(query, cos, sin, unsqueeze_dim=1)
        key = partial_rope(key, cos, sin, unsqueeze_dim=1)
        # Compact owned caches: never retain a view of Q+gate or activation buffers.
        key = key.contiguous().clone() if past is None else torch.cat((past.keys, key), 2)
        value = value.contiguous().clone() if past is None else torch.cat((past.values, value), 2)
        repeats = g.query_heads // g.kv_heads
        def repeat(tensor):
            if repeats == 1:
                return tensor
            return tensor[:, :, None].expand(batch, g.kv_heads, repeats, kv_length, g.head_dim).reshape(
                batch, g.query_heads, kv_length, g.head_dim)
        if self.attention_backend == "eager":
            scores = torch.matmul(query, repeat(key).transpose(2, 3)) * (g.head_dim**-0.5)
            probabilities = torch.softmax(scores + combined_mask, dim=-1, dtype=torch.float32).to(query.dtype)
            attended = torch.matmul(probabilities, repeat(value)).transpose(1, 2).contiguous()
        else:
            # Pinned SDPA integration repeats GQA heads when a QSA mask exists.
            # Explicit mask already encodes cache-aligned causality: do NOT also
            # request SDPA's upper-left `is_causal` triangle for cached chunks.
            attended = F.scaled_dot_product_attention(query, repeat(key), repeat(value), attn_mask=combined_mask,
                dropout_p=0.0, scale=g.head_dim**-0.5, is_causal=False).transpose(1, 2).contiguous()
            probabilities = None
        gated = attended.reshape(batch, length, -1) * torch.sigmoid(gate.reshape(batch, length, -1))
        return self.o_proj(gated), QSAState(raw, key, value), probabilities

    @classmethod
    def from_checkpoint(cls, checkpoint, geometry, prefix, *, max_weight_bytes=128 * 2**20, **budgets):
        # Meta construction avoids allocating model-sized default FP32 parameters
        # before header/dtype/budget admission. Only this one attention block loads.
        with torch.device("meta"):
            prototype = cls(geometry, **budgets)
        shapes = {name: tuple(p.shape) for name, p in prototype.named_parameters()}
        specs = {}
        for name, shape in shapes.items():
            spec = checkpoint.tensors.get(prefix + "." + name)
            if spec is None or spec.shape != shape or spec.dtype not in ("BF16", "F16", "F32"):
                raise ValueError("Original floating QSA attention weight required")
            specs[name] = spec
        if len({s.dtype for s in specs.values()}) != 1 or type(max_weight_bytes) is not int or (
            max_weight_bytes <= 0 or sum(s.nbytes for s in specs.values()) > max_weight_bytes
        ):
            raise ValueError("QSA attention weights must share dtype and fit explicit budget")
        dtype = torch_dtype(next(iter(specs.values())).dtype)
        result = prototype.to_empty(device="cpu").to(dtype=dtype)
        with torch.no_grad():
            for name, parameter in result.named_parameters():
                spec = specs[name]
                with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(spec.name)
                    if tuple(value.shape) != spec.shape or value.dtype != dtype:
                        raise ValueError("QSA backing metadata changed")
                    parameter.copy_(value)
        return result.eval()
