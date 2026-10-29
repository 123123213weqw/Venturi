"""Floating decoder-layer assembly only; no tokenizer/model/native serving.

Caller gathers PLE rows and owns NGramRequest commits separately. Layer state
is proposed out-of-place: exceptions/rejected suffixes never commit a cache.
"""

from dataclasses import dataclass

from torch import nn

from .flash_next_attention import FlashNextQSAAttention, QSAState
from .flash_next_gdn import FlashNextGDN, GDNState
from .flash_next_moe import FlashNextMoE
from .flash_next_ple import FlashNextPLE, PLEState
from .flash_next_residual import FlashNextGatedResidual


@dataclass(frozen=True)
class LayerState:
    attention: GDNState | QSAState
    ple: PLEState | None

    def snapshot(self):
        attention = self.attention.snapshot() if isinstance(self.attention, GDNState) else self.attention.truncate(
            self.attention.raw_indexer_keys.shape[1])
        return LayerState(attention, None if self.ple is None else self.ple.snapshot())


class FlashNextDecoderLayer(nn.Module):
    """PLE-before-HC, attention read/inject, MoE read/inject, private state.

    Modules must be independently bound/qualified. No full checkpoint loader or
    automatic choice of geometry/backend/placement is implied by this wrapper.
    """

    def __init__(self, attention, moe, attn_hyper_connection, mlp_hyper_connection, *, ple=None):
        super().__init__()
        if not isinstance(attention, (FlashNextGDN, FlashNextQSAAttention)) or not isinstance(moe, FlashNextMoE):
            raise ValueError("Explicit qualified floating attention/MoE required")
        h = attention.geometry.hidden_size if isinstance(attention, FlashNextGDN) else attention.geometry.indexer.hidden_size
        if any(not isinstance(m, FlashNextGatedResidual) or m.hidden_size != h or m.block_inject_weight is None
               for m in (attn_hyper_connection, mlp_hyper_connection)) or moe.geometry.hidden_size != h:
            raise ValueError("Layer attention/MoE/HC geometry or injection mismatch")
        if attn_hyper_connection.streams != mlp_hyper_connection.streams:
            raise ValueError("Layer HC stream counts must agree")
        if ple is not None and (not isinstance(ple, FlashNextPLE) or ple.geometry.hidden_size != h or
                                ple.geometry.streams != attn_hyper_connection.streams):
            raise ValueError("Layer PLE/HC geometry mismatch")
        self.attention, self.mlp = attention, moe
        self.attn_hyper_connection, self.mlp_hyper_connection, self.ple = attn_hyper_connection, mlp_hyper_connection, ple

    def forward(self, hidden_states, *, position_embeddings=None, attention_mask=None, conv_mask=None,
                ple_rows=None, past=None):
        if self.training:
            raise ValueError("Layer reference is inference-only; call eval()")
        gdn = isinstance(self.attention, FlashNextGDN)
        if past is not None and (not isinstance(past, LayerState) or
            not isinstance(past.attention, GDNState if gdn else QSAState) or (past.ple is None) != (self.ple is None)):
            raise ValueError("Layer past kind/PLE state mismatch")
        if self.ple is None and ple_rows is not None or self.ple is not None and ple_rows is None:
            raise ValueError("Layer PLE rows must match the explicit PLE configuration")
        if past is not None and past.ple is not None:
            seen = past.attention.tokens_seen if gdn else past.attention.raw_indexer_keys.shape[1]
            if past.ple.tokens_seen != seen:
                raise ValueError("PLE and attention histories must advance together")
        ple_state = None
        if self.ple is not None:
            conditioning, ple_state = self.ple(hidden_states, ple_rows, past=None if past is None else past.ple,
                                                conv_mask=conv_mask)
            hidden_states = hidden_states + conditioning
        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        attention_past = None if past is None else past.attention
        if gdn:
            output, attention_state = self.attention(mixed, past=attention_past, attention_mask=conv_mask)
        else:
            if position_embeddings is None or attention_mask is None:
                raise ValueError("QSA layer requires explicit full positions and visibility/additive mask")
            output, attention_state, _ = self.attention(mixed, position_embeddings, attention_mask, past=attention_past)
        hidden_states = self.attn_hyper_connection.inject(output, residual, injection)
        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        output = self.mlp(mixed)
        hidden_states = self.mlp_hyper_connection.inject(output, residual, injection)
        return hidden_states, LayerState(attention_state, ple_state)
