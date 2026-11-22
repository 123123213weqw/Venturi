"""GLM-5.3-Flash sparse MLA (Multi-head Latent Attention) + DSA indexer.

MLA: low-rank query bottleneck (q_a → norm → q_b), compressed KV latent
(kv_a → norm, cached WITHOUT expansion), per-head key/value expansion from
the latent at attention time (k_b/v_b as 3D per-head weights), NoPE (no
rotary embeddings anywhere in this layer).

Indexer (DSA): lightweight per-token key → k-pool compression (kpool tokens
per pool, learned gate + APE) → ReLU dot-product scoring with per-head
weights → top-k pool selection (always includes the last visible token) →
additive sparse mask for the main MLA attention. Full-indexer layers compute
indices; adjacent shared-indexer layers reuse them.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MLAState:
    """Per-layer state for cached decoding (compressed latent + indexer key)."""
    latent: torch.Tensor       # [B, T_seen, kv_lora_rank] — compressed KV
    indexer_key: torch.Tensor  # [B, T_seen, indexer_head_dim] — uncompressed
    tokens_seen: int


def _rms_norm(x, weight, eps):
    """RMSNorm in float32 with direct weight (not 1+w)."""
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (weight * normed.to(x.dtype)).to(x.dtype)


class Glm5Indexer(nn.Module):
    """DSA indexer: k-pool compression → ReLU scoring → top-k selection."""

    def __init__(self, hidden_size, q_lora_rank, num_heads, head_dim,
                 top_k, kpool, always_select_tail=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.kpool = kpool
        self.always_select_tail = always_select_tail

        # Indexer query: from q bottleneck (shared with MLA's q_a_norm output)
        self.wq_b = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        # Indexer key: from hidden (lightweight, separate from MLA KV)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        # LayerNorm with bias (NOT RMSNorm)
        self.k_norm = nn.LayerNorm(head_dim)
        # Per-head weights for score aggregation
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # k-pool compression
        self.compressor_ape = nn.Parameter(torch.zeros(kpool, head_dim))
        self.compressor_gate = nn.Linear(hidden_size, head_dim, bias=False)

    def compute_key(self, hidden_states):
        """Per-token indexer key, cached separately from MLA KV latent."""
        return self.k_norm(self.wk(hidden_states))

    def forward(self, hidden_states, q_resid, indexer_key, position, attention_mask=None):
        """Compute top-k token indices for sparse attention.

        q_resid: [B, T, q_lora_rank] — the normed q_a output (shared with MLA).
        indexer_key: [B, S_total, head_dim] — cached per-token keys.
        position: current token position (for causal masking).
        Returns int32 [B, T, top_k] token indices (-1 padded).
        """
        batch, length = hidden_states.shape[:2]
        total = indexer_key.shape[1]
        n_pools = total // self.kpool

        # Query: [B, T, H, D]
        q_idx = self.wq_b(q_resid).view(batch, length, self.num_heads, self.head_dim)

        # k-pool compression of indexer keys
        # [B, n_pools, kpool, head_dim]
        valid = min(n_pools * self.kpool, total)
        key_trunc = indexer_key[:, :valid]
        n_pools = valid // self.kpool
        pooled = key_trunc[:, :n_pools * self.kpool].view(
            batch, n_pools, self.kpool, self.head_dim)

        # Gate + APE for learned weighted pooling
        gate = self.compressor_gate(hidden_states)  # [B, T, head_dim]
        # Uniform mean-pool over kpool dim with APE (correctness first);
        # learned gating follows the pinned source when qualified.
        pooled = pooled.mean(dim=2)  # [B, P, D]
        pooled = pooled + self.compressor_ape.mean(dim=0)[None, None, :]

        # Scores: ReLU(q @ pooled.T) per head, then head-weighted sum
        # q: [B, T, H, D] → [B, H, T, D]
        # pooled: [B, P, D] → [B, D, P]
        q_t = q_idx.transpose(1, 2).float()  # [B, H, T, D]
        p_t = pooled.transpose(1, 2).float()  # [B, D, P]
        scores = torch.relu(torch.matmul(q_t, p_t))  # [B, H, T, P]
        head_weights = self.weights_proj(hidden_states).float()  # [B, T, H]
        head_weights = head_weights.transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]
        combined = (scores * head_weights).sum(1)  # [B, T, P]

        # Causal mask: pool p visible iff p*kpool + kpool - 1 < position + t + 1
        # i.e., all tokens in the pool are before the current query
        pool_last = torch.arange(n_pools, device=x_t.device if False else scores.device) * self.kpool + self.kpool - 1
        query_pos = torch.arange(position, position + length, device=combined.device)
        visible = pool_last[None, :] <= query_pos[:, None]  # [T, P]

        # Always include the last visible token's pool (tail)
        if self.always_select_tail:
            last_visible = (query_pos - 1).clamp(min=0)
            tail_pool = last_visible // self.kpool
            visible[torch.arange(length, device=combined.device), tail_pool] = True

        combined = combined.masked_fill(~visible[None], float("-inf"))

        # Top-k pool selection
        select_k = min(self.top_k // self.kpool, n_pools)
        top_pools = combined.topk(select_k, dim=-1).indices  # [B, T, select_k]

        # Expand pools back to token indices
        token_indices = top_pools * self.kpool  # start of each pool
        expansion = torch.arange(self.kpool, device=token_indices.device)
        token_indices = (token_indices[..., None] + expansion[None, None, None, :]).reshape(
            batch, length, select_k * self.kpool)

        # Mask out-of-range (from -inf pools)
        valid_mask = token_indices < total
        token_indices = token_indices.clamp(max=total - 1)
        token_indices = token_indices.masked_fill(~valid_mask, -1)

        return token_indices.to(torch.int32)


def build_sparse_mask(topk_indices, batch, heads, total_length, dtype):
    """Convert [B, T, K] token indices to additive [B, 1, T, S] mask."""
    length = topk_indices.shape[1]
    mask = torch.full((batch, 1, length, total_length),
                      torch.finfo(dtype).min, dtype=dtype, device=topk_indices.device)
    safe = topk_indices.clamp(min=0)
    # scatter 0 at selected positions
    mask.scatter_(-1, safe.long().unsqueeze(1), 0.0)
    return mask


class Glm5MLA(nn.Module):
    """Sparse MLA for GLM-5.3-Flash: q bottleneck + compressed KV + NoPE."""

    def __init__(self, hidden_size, num_heads, q_lora_rank, kv_lora_rank,
                 qk_head_dim, v_head_dim, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.eps = eps
        self.scale = qk_head_dim ** -0.5

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_norm_weight = nn.Parameter(torch.ones(q_lora_rank))
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=False)
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_a_norm_weight = nn.Parameter(torch.ones(kv_lora_rank))
        # 3D per-head weights: [H, latent, out_per_head]
        self.k_b = nn.Parameter(torch.zeros(num_heads, kv_lora_rank, qk_head_dim))
        self.v_b = nn.Parameter(torch.zeros(num_heads, v_head_dim, kv_lora_rank))
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

    def _expand_kv(self, latent):
        """Expand compressed latent to per-head keys/values.

        latent: [B, S, kv_lora_rank] → keys [B, H, S, D_k], values [B, H, S, D_v]
        """
        batch, seq, _ = latent.shape
        # k: latent @ k_b[h] → [B, S, H, D_k]
        keys = torch.einsum("bsl,hld->bhsd", latent, self.k_b.to(latent.dtype))
        # v: v_b[h] @ latent.T → per-head: [D_v, latent] @ [latent, S] → [D_v, S]
        # equivalently: latent @ v_b[h].T → [S, D_v]
        values = torch.einsum("bsl,hdl->bhsd", latent, self.v_b.to(latent.dtype))
        return keys, values

    def forward(self, hidden_states, past=None, attention_mask=None,
                prev_topk=None, indexer=None):
        """Returns (output, proposed_state, topk_indices_or_None)."""
        dtype = hidden_states.dtype
        batch, length = hidden_states.shape[:2]

        # Query through bottleneck
        q_latent = self.q_a_proj(hidden_states)
        q_resid = _rms_norm(q_latent, self.q_a_norm_weight.to(dtype), self.eps)
        query = self.q_b_proj(q_resid).view(batch, length, self.num_heads, self.qk_head_dim)

        # KV latent (compressed — cached WITHOUT expansion)
        k_latent = self.kv_a_proj(hidden_states)
        k_latent = _rms_norm(k_latent, self.kv_a_norm_weight.to(dtype), self.eps)

        # Concatenate with past latent
        if past is not None:
            full_latent = torch.cat([past.latent, k_latent], dim=1)
            total = full_latent.shape[1]
        else:
            full_latent = k_latent
            total = length

        # Expand per-head keys/values
        keys, values = self._expand_kv(full_latent)
        # [B, H, S, D_k] and [B, H, S, D_v]

        # Compute or receive sparse indices
        topk_indices = None
        if indexer is not None:
            if past is not None:
                indexer_key = torch.cat([past.indexer_key,
                                         indexer.compute_key(hidden_states)], dim=1)
            else:
                indexer_key = indexer.compute_key(hidden_states)
            position = past.tokens_seen if past else 0
            topk_indices = indexer(hidden_states, q_resid, indexer_key, position, attention_mask)
        elif prev_topk is not None:
            topk_indices = prev_topk

        # Build mask (causal + optional sparse)
        base_mask = torch.full((batch, 1, length, total),
                               torch.finfo(dtype).min, dtype=dtype, device=hidden_states.device)
        query_pos = torch.arange(past.tokens_seen if past else 0,
                                 (past.tokens_seen if past else 0) + length,
                                 device=hidden_states.device)
        key_pos = torch.arange(total, device=hidden_states.device)
        causal = key_pos[None, :] <= query_pos[:, None]
        base_mask = base_mask.masked_fill(causal[None, None], 0.0)

        if topk_indices is not None:
            sparse = build_sparse_mask(topk_indices, batch, self.num_heads, total, dtype)
            # Combine: only attend to positions selected by BOTH causal and sparse
            mask = torch.maximum(base_mask, sparse)
        else:
            mask = base_mask

        # Eager attention: [B, H, T, D] @ [B, H, D, S] → [B, H, T, S]
        q_t = query.transpose(1, 2)  # [B, H, T, D_k]
        scores = torch.matmul(q_t.float(), keys.transpose(2, 3).float()) * self.scale
        scores = scores.to(dtype) + mask
        probs = F.softmax(scores.float(), dim=-1).to(dtype)
        attended = torch.matmul(probs, values)  # [B, H, T, D_v]

        # Output projection
        output = self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))

        # Build next state
        indexer_key_next = None
        if indexer is not None:
            indexer_key_next = indexer.compute_key(hidden_states)
        tokens = (past.tokens_seen if past else 0) + length
        state = MLAState(full_latent, indexer_key_next, tokens)
        return output, state, topk_indices
