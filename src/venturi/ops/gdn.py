"""Gated DeltaNet recurrence, not a complete Qwen hybrid-model implementation.

q/k [T,Hqk,Dk], v/out [T,Hv,Dv], log_decay/beta [T,Hv]. State is
[Hv,Dv,Dk], FP32 natively. For each token/head:
  A = exp(log_decay) * state
  delta = beta * (v - A @ k)
  state = A + outer(delta, k)
  out = scale * state @ q
Optional q/k L2 normalization uses FP32 with epsilon 1e-6, without an activation-dtype
normalization boundary. Gates are already transformed (not raw a/b projections). A packed convolution
output may be consumed as [T, Q-flat | K-flat | V-flat] without unpack copies.
The caller owns/reset/commits state; there is no hidden cursor or cache manager.
"""

from dataclasses import dataclass
import math

import torch

from .blocks import _DTYPES, _no_overlap, _shape, _tensor


def prepare_gdn(
    backend,
    *,
    tokens,
    qk_heads,
    value_heads,
    key_dim=128,
    value_dim=128,
    normalize_qk=True,
    scale=None,
    dtype=torch.bfloat16,
    device="cuda:0",
):
    _shape(tokens, qk_heads)
    _shape(value_heads, value_dim)
    if type(key_dim) is not int or key_dim not in (64, 128, 256):
        raise ValueError("GDN key_dim must be 64, 128 or 256")
    if value_heads % qk_heads or value_heads * value_dim >= 2**31:
        raise ValueError(
            "GDN value_heads must be divisible by qk_heads; rows fit int32"
        )
    if type(normalize_qk) is not bool:
        raise ValueError("normalize_qk must be boolean")
    scale = 1 / math.sqrt(key_dim) if scale is None else scale
    if (
        type(scale) not in (int, float)
        or not math.isfinite(scale)
        or not torch.finfo(torch.float32).tiny
        <= scale
        <= torch.finfo(torch.float32).max
    ):
        raise ValueError("GDN scale must be a normal positive FP32 value")
    if dtype not in (torch.float16, torch.bfloat16) and not (
        backend.execution_kind == "torch" and dtype == torch.float64
    ):
        raise ValueError("GDN supports FP16/BF16, plus explicit FP64 reference mode")
    return PreparedGDN(
        backend,
        tokens,
        qk_heads,
        value_heads,
        key_dim,
        value_dim,
        normalize_qk,
        float(scale),
        dtype,
        backend.validate_device(dtype, device),
    )


@dataclass(frozen=True)
class PreparedGDN:
    backend: object
    tokens: int
    qk_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    normalize_qk: bool
    scale: float
    dtype: torch.dtype
    device: torch.device
    _native_entry: str = "venturi_gdn_run"

    def run(self, q, k, v, log_decay, beta, *, state, out, state_out):
        """Exact state/state_out alias allowed; all other writable aliases rejected.

        Inputs must be finite and gates have the intended stable semantics
        (log_decay <= 0, beta in [0,1]). No implicit sigmoid/softplus, gate clamp,
        device synchronization, or numerical scan occurs during native run().
        A failed asynchronous execution invalidates caller-owned state.
        """
        for name, tensor, shape in (
            ("q", q, (self.tokens, self.qk_heads, self.key_dim)),
            ("k", k, (self.tokens, self.qk_heads, self.key_dim)),
            ("v", v, (self.tokens, self.value_heads, self.value_dim)),
        ):
            _tensor(tensor, shape, self.dtype, self.device, name)
        return self._run(q, k, v, log_decay, beta, state, out, state_out, packed=False)

    @property
    def packed_features(self):
        return 2 * self.qk_heads * self.key_dim + self.value_heads * self.value_dim

    def run_packed(self, qkv, log_decay, beta, *, state, out, state_out):
        """Consume contiguous [T, Q-flat | K-flat | V-flat], with no unpack copy.

        The token stride is the full packed width. Heads use the same contiguous
        grouping and FP32 recurrence as run(). The complete packed input is
        checked for writable overlap, including K/V sections and later tokens.
        This is one sequence, not a batch of independent request states.
        """
        _tensor(
            qkv, (self.tokens, self.packed_features), self.dtype, self.device, "qkv"
        )
        return self._run(
            qkv, None, None, log_decay, beta, state, out, state_out, packed=True
        )

    def _run(self, q, k, v, log_decay, beta, state, out, state_out, *, packed):
        gate_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
        for name, tensor, shape, dtype in (
            ("log_decay", log_decay, (self.tokens, self.value_heads), gate_dtype),
            ("beta", beta, (self.tokens, self.value_heads), gate_dtype),
            (
                "state",
                state,
                (self.value_heads, self.value_dim, self.key_dim),
                gate_dtype,
            ),
            (
                "state_out",
                state_out,
                (self.value_heads, self.value_dim, self.key_dim),
                gate_dtype,
            ),
            ("out", out, (self.tokens, self.value_heads, self.value_dim), self.dtype),
        ):
            _tensor(tensor, shape, dtype, self.device, name)
        readonly = (q, log_decay, beta) if packed else (q, k, v, log_decay, beta)
        _no_overlap(state_out, readonly + (out,))
        _no_overlap(state_out, (state,), allow_inplace=True)
        _no_overlap(out, readonly + (state,))
        if self.backend.execution_kind == "torch":
            if packed:
                width = self.qk_heads * self.key_dim
                q, k, v = (
                    q[:, :width].reshape(self.tokens, self.qk_heads, self.key_dim),
                    q[:, width : 2 * width].reshape(
                        self.tokens, self.qk_heads, self.key_dim
                    ),
                    q[:, 2 * width :].reshape(
                        self.tokens, self.value_heads, self.value_dim
                    ),
                )
            work = torch.float64 if self.dtype == torch.float64 else torch.float32
            qs, ks = q.to(work), k.to(work)
            if self.normalize_qk:
                qs = qs * torch.rsqrt(qs.square().sum(-1, keepdim=True) + 1e-6)
                ks = ks * torch.rsqrt(ks.square().sum(-1, keepdim=True) + 1e-6)
            # Explicit allocating reference, never a native fallback.
            s = state.clone()
            group = self.value_heads // self.qk_heads
            for t in range(self.tokens):
                qt, kt = (
                    qs[t].repeat_interleave(group, 0),
                    ks[t].repeat_interleave(group, 0),
                )
                s = s * log_decay[t].exp()[:, None, None]
                delta = beta[t, :, None] * (
                    v[t].to(work) - (s * kt[:, None, :]).sum(-1)
                )
                s = s + delta[:, :, None] * kt[:, None, :]
                out[t].copy_(((s * qt[:, None, :]).sum(-1) * self.scale).to(self.dtype))
            state_out.copy_(s)
        else:
            if packed:
                width_bytes = self.qk_heads * self.key_dim * q.element_size()
                qkv_ptrs = (
                    q.data_ptr(),
                    q.data_ptr() + width_bytes,
                    q.data_ptr() + 2 * width_bytes,
                )
            else:
                qkv_ptrs = (q.data_ptr(), k.data_ptr(), v.data_ptr())
            with torch.cuda.device(self.device):
                self.backend._check(
                    getattr(self.backend._native, self._native_entry)(
                        *qkv_ptrs,
                        *[
                            x.data_ptr()
                            for x in (log_decay, beta, state, out, state_out)
                        ],
                        self.tokens,
                        self.qk_heads,
                        self.value_heads,
                        self.key_dim,
                        self.value_dim,
                        int(self.normalize_qk),
                        int(packed),
                        _DTYPES[self.dtype],
                        self.scale,
                        torch.cuda.current_stream(self.device).cuda_stream,
                    )
                )
        return out
