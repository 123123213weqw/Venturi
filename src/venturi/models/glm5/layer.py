"""GLM-5.3-Flash decoder layer: mHC → norm → attention → inject → mHC → norm → MoE → inject."""

import torch
import torch.nn.functional as F
from torch import nn

from venturi.models.glm5.hc import Glm5HyperConnection
from venturi.models.glm5.kda import Glm5KDA, KDAState
from venturi.models.glm5.mla import Glm5MLA, MLAState


class DenseMLP(nn.Module):
    """Dense feed-forward for the first 3 blocks (SwiGLU with clamp)."""

    def __init__(self, hidden_size, intermediate_size, swiglu_limit=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


def _rms(x, weight, eps):
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (weight * normed.to(x.dtype)).to(x.dtype)


class Glm5DecoderLayer(nn.Module):
    """One decoder block. Attention is KDA or MLA (polymorphic); MLP is MoE or dense."""

    def __init__(self, hidden_size, streams, eps=1e-5, hc_eps=1e-6, hc_iters=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.attn_hc = Glm5HyperConnection(streams, hidden_size, hc_iters, hc_eps, eps)
        self.ffn_hc = Glm5HyperConnection(streams, hidden_size, hc_iters, hc_eps, eps)
        self.attn_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.ffn_norm_weight = nn.Parameter(torch.ones(hidden_size))
        # Sub-modules assigned externally (attention: Glm5KDA|Glm5MLA; mlp: DenseMLP|MoE)
        self.attention = None
        self.mlp = None

    def _rms(self, x, weight):
        return _rms(x, weight.to(x.dtype), self.eps)

    def forward(self, hidden_streams, past=None, attention_mask=None, prev_topk=None):
        """Returns (hidden_streams, layer_state, topk_indices)."""
        # --- Attention sub-layer ---
        post_a, comb_a, collapsed = self.attn_hc(hidden_streams)
        normed = self._rms(collapsed, self.attn_norm_weight)
        if isinstance(self.attention, Glm5KDA):
            attn_out, attn_state = self.attention(normed, past=past, attention_mask=attention_mask)
            topk = None
        else:
            indexer = getattr(self.attention, '_indexer', None)
            attn_out, attn_state, topk = self.attention(
                normed, past=past, attention_mask=attention_mask,
                prev_topk=prev_topk, indexer=indexer)
        hidden_streams = self.attn_hc.inject(attn_out, hidden_streams, post_a, comb_a)

        # --- MLP sub-layer ---
        post_f, comb_f, collapsed = self.ffn_hc(hidden_streams)
        normed = self._rms(collapsed, self.ffn_norm_weight)
        mlp_out = self.mlp(normed)
        hidden_streams = self.ffn_hc.inject(mlp_out, hidden_streams, post_f, comb_f)

        return hidden_streams, attn_state, topk
