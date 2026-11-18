"""Kimi Delta Attention (KDA) for GLM-5.3-Flash linear-attention layers.

Frontend: separate q/k/v projections concatenated into one fused depthwise
causal conv1d (the GGUF stores three conv weight slices that are concatenated
at load time, matching the transformers module); low-rank sigmoid-bounded
forget gate (f_a → f_b bottleneck with dt_bias and A_log); low-rank output
gate (g_a → g_b bottleneck); beta input gate; sigmoid-gated RMS output norm.

Core: the shared gated_delta_rule (L2-normalized q/k, FP32 state
[B, heads, head_dim, head_dim], chunked prefill and recurrent decode).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from venturi.models.linear_attention import SigmoidGatedRMSNorm


@dataclass(frozen=True)
class KDAState:
    """Per-layer recurrent state for cached decoding."""
    conv: torch.Tensor            # [B, 3*qkv_dim, conv_kernel] raw projection history
    recurrent: torch.Tensor       # [B, heads, head_dim, head_dim] FP32
    tokens_seen: int




def _kda_delta_reference(query, key, value, g, beta, *, initial_state=None):
    """Per-token KDA reference with per-key-dim decay (GLM KDA semantics).

    Unlike the flash-next GDN scalar decay, GLM KDA's forget gate produces a
    [B,T,H,D] log-decay applied per key dimension. Reference implementation
    in float32; chunked optimization is future work.
    """
    dtype = query.dtype
    q = query.float()
    k = key.float()
    v = value.float()
    # L2 normalize q/k (eps inside rsqrt), then 1/sqrt(dk) scale
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    dk = q.shape[-1]
    q = q * (dk ** -0.5)

    batch, length, heads, dim = q.shape
    if initial_state is None:
        state = v.new_zeros(batch, heads, dim, v.shape[-1])
    else:
        state = initial_state.float().clone()

    output = torch.empty(batch, length, heads, v.shape[-1], dtype=torch.float32, device=v.device)
    for t in range(length):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        gt = g[:, t].float().exp()          # [B, H, D] positive decay
        bt = beta[:, t].float()             # [B, H]
        state = state * gt.unsqueeze(-1)    # [B,H,D,1] broadcast over value dim
        memory = (state * kt.unsqueeze(-1)).sum(-2)
        correction = (vt - memory) * bt.unsqueeze(-1)
        state = state + kt.unsqueeze(-1) * correction.unsqueeze(-2)
        output[:, t] = (state * qt.unsqueeze(-1)).sum(-2)
    return output.to(dtype), state

class Glm5KDA(nn.Module):
    """GLM-5.3-Flash KDA linear attention (reference implementation)."""

    def __init__(self, hidden_size, num_heads, head_dim, conv_kernel,
                 gate_lower_bound=-5.0, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_dim = num_heads * head_dim
        self.conv_kernel = conv_kernel
        self.gate_lower_bound = gate_lower_bound
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.qkv_dim * 3, self.qkv_dim * 3, kernel_size=conv_kernel,
            groups=self.qkv_dim * 3, padding=conv_kernel - 1, bias=False,
        )
        # Forget gate: low-rank bottleneck through head_dim
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim))
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        # Beta gate
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # Output gate: low-rank bottleneck through head_dim
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.qkv_dim, bias=False)
        # Output norm and projection
        self.o_norm = SigmoidGatedRMSNorm(head_dim, eps=eps)
        self.o_proj = nn.Linear(self.qkv_dim, hidden_size, bias=False)

    def _forget_gate(self, hidden_states):
        """Sigmoid-bounded log-decay: g = lower_bound * sigmoid(exp(A_log) * raw)."""
        hidden_shape = (*hidden_states.shape[:2], self.num_heads, self.head_dim)
        raw = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (raw.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)
        # Sigmoid-bounded: g in (lower_bound, 0), always negative log-decay
        return self.gate_lower_bound * torch.sigmoid(decay_rate * g)

    def forward(self, hidden_states, past=None, attention_mask=None):
        """Returns (output, proposed_state). No state mutation."""
        if hidden_states.dtype != self.q_proj.weight.dtype:
            raise ValueError("KDA input dtype must match the module dtype")
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None]
        batch, length = hidden_states.shape[:2]
        hidden_shape = (batch, length, self.num_heads, self.head_dim)

        qkv = torch.cat([
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        ], dim=-1).transpose(1, 2)  # [B, 3*qkv_dim, T]

        # Conv state management (matching GDN's history pattern)
        # Causal depthwise conv: always left-pad with (kernel-1) history
        # tokens (from state or zeros), apply conv with padding=0, take all
        # valid outputs (which is exactly `length` new positions).
        flat_weight = self.conv1d.weight  # [3*qkv_dim, 1, kernel]
        if past is not None:
            padded = torch.cat([past.conv, qkv], dim=-1)
        else:
            padded = F.pad(qkv, (self.conv_kernel - 1, 0))
        convolved = F.conv1d(padded, flat_weight, None, 1, 0,
                             groups=self.qkv_dim * 3)
        proposed_conv = padded[..., -self.conv_kernel:]
        convolved = F.silu(convolved)[..., -length:]
        qkv_out = convolved.transpose(1, 2)
        query, key, value = qkv_out.split([self.qkv_dim] * 3, dim=-1)
        query = query.reshape(hidden_shape)
        key = key.reshape(hidden_shape)
        value = value.reshape(hidden_shape)

        g = self._forget_gate(hidden_states)       # [B, T, H, D] negative log-decay
        beta = torch.sigmoid(self.b_proj(hidden_states))  # [B, T, H]

        core_out, proposed_recurrent = _kda_delta_reference(
            query, key, value, g, beta,
            initial_state=None if past is None else past.recurrent,
        )

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(hidden_shape)
        normed = self.o_norm(core_out.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = self.o_proj(normed.reshape(batch, length, -1))
        tokens = (past.tokens_seen if past else 0) + length
        return output, KDAState(proposed_conv, proposed_recurrent, tokens)
