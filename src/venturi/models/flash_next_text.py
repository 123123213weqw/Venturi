"""Bounded floating text-core reference; not a full checkpoint/native engine.

Plain contiguous text positions and explicit visibility only. No multimodal or
scaled/dynamic RoPE, paging, scheduler, graph, speculation policy or speed claim.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .flash_next_attention import FlashNextQSAAttention
from .flash_next_gdn import GDNState
from .flash_next_layer import LayerState
from .flash_next_layer import FlashNextDecoderLayer
from .flash_next_residual import FlashNextGatedResidual


@dataclass(frozen=True)
class TextState:
    layers: tuple
    visible_keys: torch.Tensor
    tokens_seen: int

    def snapshot(self):
        return TextState(tuple(s.snapshot() for s in self.layers), self.visible_keys.clone(), self.tokens_seen)


class FlashNextTextCore(nn.Module):
    """Embedding -> repeated HC vectors -> layers -> final mixer -> output head.

    Modules are already explicitly bound. PLE rows are an explicit per-layer
    mapping, not hidden I/O. A request wrapper must coordinate token-hash state.
    Every call returns proposed out-of-place cache; it never commits a request.
    """

    def __init__(self, embed_tokens, layers, final_mixer, lm_head, *, rotary_dim,
                 rope_theta=10000000.0, max_sequence=4096, max_mask_elements=4*2**20,
                 max_logits_elements=4*2**20):
        super().__init__()
        layers = tuple(layers)
        if not isinstance(embed_tokens, nn.Embedding) or not layers or len(layers) > 64 or any(
                not isinstance(layer, FlashNextDecoderLayer) for layer in layers):
            raise ValueError("Explicit embedding and bounded qualified decoder layers required")
        h, vocab = embed_tokens.embedding_dim, embed_tokens.num_embeddings
        if not isinstance(final_mixer, FlashNextGatedResidual) or final_mixer.block_inject_weight is not None or (
                final_mixer.hidden_size != h) or not isinstance(lm_head, nn.Linear) or (
                lm_head.in_features != h or lm_head.out_features != vocab or lm_head.bias is not None):
            raise ValueError("Explicit final non-injecting mixer and bias-free matching output head required")
        if any(layer.attn_hyper_connection.hidden_size != h or
               layer.attn_hyper_connection.streams != final_mixer.streams for layer in layers):
            raise ValueError("Text layer HC geometry mismatch")
        if type(rotary_dim) is not int or not 2 <= rotary_dim <= 256 or rotary_dim % 2 or isinstance(rope_theta, bool) or (
                not isinstance(rope_theta, (int, float)) or not math.isfinite(rope_theta) or rope_theta <= 0):
            raise ValueError("Explicit even default rotary width and positive finite theta required")
        if any(type(n) is not int or not 0 < n <= 64*2**20 for n in
               (max_sequence, max_mask_elements, max_logits_elements)):
            raise ValueError("Bounded text sequence/mask/logits admission required")
        for layer in layers:
            if isinstance(layer.attention, FlashNextQSAAttention) and (
                rotary_dim > min(layer.attention.geometry.head_dim, layer.attention.geometry.indexer.head_dim)):
                raise ValueError("Shared rotary width exceeds QSA/indexer head width")
        self.embed_tokens, self.layers = embed_tokens, nn.ModuleList(layers)
        self.hyper_connection_mixer, self.lm_head = final_mixer, lm_head
        self.rotary_dim, self.rope_theta = rotary_dim, rope_theta
        self.max_sequence, self.max_mask_elements, self.max_logits_elements = (
            max_sequence, max_mask_elements, max_logits_elements)
        # Source constructs default inverse frequencies on CPU in FP32. Keep
        # this immutable reference value out of module.to(dtype) weight casts.
        self._inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu") / rotary_dim))

    @property
    def ple_layers(self):
        return tuple(i for i, layer in enumerate(self.layers) if layer.ple is not None)

    def position_embeddings(self, batch, total, *, device, dtype):
        positions = torch.arange(total, device=device).view(1, -1).expand(batch, -1)
        freq = positions[..., None].float() * self._inv_freq.to(device=device, dtype=torch.float32)
        cos, sin = freq.cos(), freq.sin()
        return torch.cat((cos, cos), -1).to(dtype), torch.cat((sin, sin), -1).to(dtype)

    def forward(self, input_ids, *, ple_rows=None, conv_mask=None, past=None):
        if self.training:
            raise ValueError("Text reference is inference-only; call eval()")
        weight = self.embed_tokens.weight
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.device != weight.device or (
                min(input_ids.shape) < 1) or (input_ids < 0).any() or (input_ids >= self.embed_tokens.num_embeddings).any():
            raise ValueError("Nonempty device-local 2D I64 vocabulary token IDs required")
        batch, length = input_ids.shape
        seen = 0 if past is None else past.tokens_seen if isinstance(past, TextState) else -1
        total = seen + length
        if type(seen) is not int or seen < 0 or total > self.max_sequence or batch*length*total > self.max_mask_elements or (
                batch*length*self.lm_head.out_features > self.max_logits_elements):
            raise ValueError("Text sequence/mask/logits budget exceeded or invalid state")
        if conv_mask is not None and (conv_mask.shape != input_ids.shape or conv_mask.dtype != torch.bool or
                                      conv_mask.device != input_ids.device):
            raise ValueError("Current conv mask must be device-local 2D bool")
        rows = {} if ple_rows is None else ple_rows
        if not isinstance(rows, dict) or set(rows) != set(self.ple_layers):
            raise ValueError("Explicit PLE rows must cover exactly the configured layers")
        if past is not None:
            if len(past.layers) != len(self.layers) or past.visible_keys.shape != (batch, seen) or (
                past.visible_keys.dtype != torch.bool or past.visible_keys.device != input_ids.device):
                raise ValueError("Text past layer/visibility geometry mismatch")
            for s in past.layers:
                if not isinstance(s, LayerState):
                    raise ValueError("Text past requires explicit layer states")
                count = s.attention.tokens_seen if isinstance(s.attention, GDNState) else s.attention.raw_indexer_keys.shape[1]
                if count != seen:
                    raise ValueError("All layer histories must advance together")
        visible = torch.ones_like(input_ids, dtype=torch.bool) if conv_mask is None else conv_mask
        visible = visible.clone() if past is None else torch.cat((past.visible_keys, visible), -1)
        causal = torch.arange(total, device=weight.device)[None, None, None, :] <= torch.arange(
            seen, total, device=weight.device)[None, None, :, None]
        attention_mask = causal & visible[:, None, None, :]
        hidden = self.embed_tokens(input_ids)
        position_embeddings = self.position_embeddings(batch, total, device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.repeat(1, 1, self.hyper_connection_mixer.streams)
        states = []
        for i, layer in enumerate(self.layers):
            hidden, proposed = layer(hidden, position_embeddings=position_embeddings, attention_mask=attention_mask,
                conv_mask=conv_mask, ple_rows=rows.get(i), past=None if past is None else past.layers[i])
            states.append(proposed)
        hidden = self.hyper_connection_mixer(hidden)
        return self.lm_head(hidden), TextState(tuple(states), visible, total)
