"""GLM-5.3-Flash text model: embedding → 45 layers + MTP → mean(head) → norm → lm_head."""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from venturi.models.glm5.geometry import Glm5TextGeometry, glm5_text_geometry_from_metadata
from venturi.models.glm5.hc import Glm5HyperConnection
from venturi.models.glm5.kda import Glm5KDA
from venturi.models.glm5.layer import DenseMLP, Glm5DecoderLayer, _rms
from venturi.models.glm5.mla import Glm5MLA, Glm5Indexer


@dataclass
class ModelState:
    """Per-layer states for cached decoding."""
    layers: list = field(default_factory=list)
    tokens_seen: int = 0


class Glm5TextModel(nn.Module):
    """Full GLM-5.3-Flash text model (reference implementation)."""

    def __init__(self, geometry: Glm5TextGeometry):
        super().__init__()
        self.geometry = geometry
        g = geometry
        self.embed = nn.Embedding(g.vocab_size, g.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(g.num_layers + g.nextn_layers):
            layer = Glm5DecoderLayer(g.hidden_size, g.hc_streams,
                                      g.rms_eps, g.hc_eps, g.hc_sinkhorn_iterations)
            if g.attention_kind(i) == "sparse_mla":
                layer.attention = Glm5MLA(g.hidden_size, g.head_count,
                                           g.q_lora_rank, g.kv_lora_rank,
                                           g.key_length_mla, g.value_length_mla, g.rms_eps)
            else:
                layer.attention = Glm5KDA(g.hidden_size, g.kda_heads, g.kda_head_dim,
                                           g.ssm_conv_kernel, g.gate_lower_bound, g.rms_eps)
            if g.mlp_kind(i) == "dense":
                layer.mlp = DenseMLP(g.hidden_size, g.dense_feed_forward)
            else:
                layer.mlp = None  # assigned externally (universal MoE)
            self.layers.append(layer)
        self.output_norm_weight = nn.Parameter(torch.ones(g.hidden_size))
        self.lm_head = nn.Linear(g.hidden_size, g.vocab_size, bias=False)

    def forward(self, input_ids, past=None, attention_mask=None):
        """Returns (logits [B,T,vocab], new_state)."""
        g = self.geometry
        batch, length = input_ids.shape
        hidden = self.embed(input_ids)
        # Expand to streams: [B,T,N,D]
        hidden_streams = hidden.unsqueeze(2).expand(batch, length, g.hc_streams, g.hidden_size)

        states = []
        topk = None
        start = past.tokens_seen if past else 0
        for i, layer in enumerate(self.layers):
            layer_past = past.layers[i] if past else None
            hidden_streams, state, topk = layer(
                hidden_streams, past=layer_past,
                attention_mask=attention_mask, prev_topk=topk)
            states.append(state)

        # Final: mean over streams → output norm → lm_head
        collapsed = hidden_streams.mean(dim=2)  # [B,T,D]
        collapsed = _rms(collapsed, self.output_norm_weight.to(collapsed.dtype), g.rms_eps)
        logits = self.lm_head(collapsed)
        return logits, ModelState(states, start + length)
