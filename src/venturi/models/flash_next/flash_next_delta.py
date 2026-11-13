"""Floating gated-delta core and sigmoid-gated RMS for Flash-Next qualification.

Explicit FP32 state layout [B, value_heads, key_dim, value_dim]. Callers must
expand K heads to V heads first (official16->48), and keep convolution state
separately. Not a full GDN frontend/native kernel/automatic backend selector.
Chunk and recurrent evaluation have different rounding and are not bit-equal.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SigmoidGatedRMSNorm(nn.Module):
    """Direct RMS weight, NOT zero-centered1+w; normalize/cast before gate."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        if type(width) is not int or width < 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Positive gated RMS width/epsilon required")
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, values, gate):
        if values.shape != gate.shape or values.shape[-1] != self.weight.numel() or any(
            t.device != self.weight.device or t.dtype != self.weight.dtype for t in (values, gate)
        ):
            raise ValueError("GDN gate/value/weight geometry, dtype and device must match")
        normed = values.float() * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + self.eps)
        scaled = self.weight * normed.to(values.dtype)
        return (scaled * torch.sigmoid(gate.float())).to(values.dtype)


def gated_delta_rule(query, key, value, decay, beta, *, initial_state=None, mode="recurrent",
                     normalize_qk=True, chunk_size=64, max_work_elements=16 * 2**20):
    """Same-mode pinned floating semantics with bounded dense temporary storage.

Input Q/K [B,T,HV,DK], V [B,T,HV,DV], decay/beta [B,T,HV].
No input/past mutation; returns (output in input dtype, proposed FP32 state).
`decay` is log-decay, not exp(decay). This is a reference implementation:
Python scans and dense triangular solves are not V100 optimization claims.
"""
    if mode not in ("recurrent", "chunk") or type(normalize_qk) is not bool:
        raise ValueError("Explicit recurrent/chunk mode and boolean normalization required")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 256 or type(max_work_elements) is not int or (
        not 0 < max_work_elements <= 64 * 2**20
    ):
        raise ValueError("Explicit bounded chunk/work budget required")
    if query.ndim != 4 or query.shape != key.shape or value.ndim != 4 or value.shape[:3] != query.shape[:3] or (
        min(query.shape) < 1 or value.shape[-1] < 1 or decay.shape != query.shape[:3] or beta.shape != decay.shape
    ):
        raise ValueError("GDN expanded Q/K/V/decay/beta shapes mismatch")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16) or any(
        t.device != query.device for t in (key, value, decay, beta)
    ) or any(t.dtype != query.dtype for t in (key, value, beta)) or decay.dtype not in (query.dtype, torch.float32):
        raise ValueError("GDN floating input dtype/device mismatch")
    batch, length, heads, dk = query.shape
    dv = value.shape[-1]
    state_shape = (batch, heads, dk, dv)
    if initial_state is not None and (initial_state.shape != state_shape or initial_state.dtype != torch.float32 or
                                      initial_state.device != query.device):
        raise ValueError("GDN state must be FP32 [B,HV,DK,DV] on the input device")
    padded_length = length if mode == "recurrent" else ((length + chunk_size - 1) // chunk_size) * chunk_size
    # A conservative element estimate for state, input/output and pairwise chunk
    # intermediates, not a promise of allocator/RSS usage or full-model capacity.
    estimate = batch * heads * (dk * dv + padded_length * (4 * dk + 4 * dv + 4))
    if mode == "chunk":
        estimate += batch * heads * padded_length * chunk_size * 4
    if estimate > max_work_elements:
        raise ValueError("GDN reference exceeds explicit work budget")
    dtype = query.dtype
    q, k, v, b, g = (t.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                     for t in (query, key, value, beta, decay))
    if normalize_qk:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    state = v.new_zeros(state_shape) if initial_state is None else initial_state
    if mode == "recurrent":
        q = q / (dk**0.5)
        output = torch.zeros_like(v)
        for t in range(length):
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
            state = state * g[:, :, t].exp()[..., None, None]
            memory = (state * kt.unsqueeze(-1)).sum(-2)
            correction = (vt - memory) * b[:, :, t].unsqueeze(-1)
            state = state + kt.unsqueeze(-1) * correction.unsqueeze(-2)
            output[:, :, t] = (state * qt.unsqueeze(-1)).sum(-2)
        return output.transpose(1, 2).contiguous().to(dtype), state
    q = q * (dk**-0.5)
    pad = padded_length - length
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    b, g = (F.pad(t, (0, pad)) for t in (b, g))
    kb, vb = k * b.unsqueeze(-1), v * b.unsqueeze(-1)
    q, k, kb, vb = (t.reshape(batch, heads, -1, chunk_size, t.shape[-1]) for t in (q, k, kb, vb))
    cumulative = g.reshape(batch, heads, -1, chunk_size).cumsum(3)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    pairwise = (cumulative.unsqueeze(4) - cumulative.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    triangular = (kb @ k.transpose(-1, -2)) * pairwise
    local_attention = (q @ k.transpose(-1, -2)) * pairwise
    decayed_kb = kb * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(triangular, vb, upper=False, unitriangular=True)
    past_reads = torch.linalg.solve_triangular(triangular, decayed_kb, upper=False, unitriangular=True)
    output = torch.zeros_like(new_values)
    q = q * cumulative.exp().unsqueeze(-1)
    k = k * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    for t in range(padded_length // chunk_size):
        corrections = new_values[:, :, t] - past_reads[:, :, t] @ state
        output[:, :, t] = q[:, :, t] @ state + local_attention[:, :, t] @ corrections
        state = state * chunk_decay[:, :, t] + k[:, :, t].transpose(-1, -2) @ corrections
    return output.reshape(batch, heads, -1, dv)[:, :, :length].transpose(1, 2).to(
        dtype, memory_format=torch.contiguous_format), state
