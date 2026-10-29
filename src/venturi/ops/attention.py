"""Explicit RoPE and causal GQA operators, independent of model configuration.

Native attention is an online-softmax correctness baseline, not FlashAttention,
PagedAttention, or a KV-cache manager. Layout is contiguous token/head/channel.
No implicit projection, frequency generation, cache append, or table selection.
"""

from dataclasses import dataclass
import math

import torch

from .blocks import _DTYPES, _no_overlap, _shape, _tensor


def _dimension(value, name):
    if type(value) is not int or not 0 < value < 2**31:
        raise ValueError(f"{name} must be a positive int32")
    return value


def prepare_rope(
    backend,
    *,
    tokens,
    heads,
    head_dim,
    rotary_dim=None,
    layout="split_half",
    dtype,
    device,
):
    _shape(tokens, heads)
    _dimension(head_dim, "head_dim")
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    _dimension(rotary_dim, "rotary_dim")
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and at most head_dim")
    if layout not in ("split_half", "interleaved"):
        raise ValueError("layout must be split_half or interleaved")
    target = backend.validate_device(dtype, device)
    return PreparedRoPE(
        backend, tokens, heads, head_dim, rotary_dim, layout, dtype, target
    )


def prepare_attention(
    backend,
    *,
    tokens,
    kv_tokens,
    query_heads,
    kv_heads,
    head_dim,
    dtype,
    device,
    scale=None,
):
    _shape(tokens, kv_tokens)
    for name, value in (
        ("query_heads", query_heads),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
    ):
        _dimension(value, name)
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if kv_tokens < tokens:
        raise ValueError("kv_tokens must include all query tokens")
    if head_dim not in (64, 128, 256):
        raise ValueError("attention currently qualifies head_dim 64, 128 or 256")
    if tokens * query_heads >= 2**31:
        raise ValueError("query row count must fit int32")
    scale = 1 / math.sqrt(head_dim) if scale is None else scale
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    # Native arithmetic is FP32; avoid silently underflowing/overflowing a scale.
    if not torch.finfo(torch.float32).tiny <= scale <= torch.finfo(torch.float32).max:
        raise ValueError("scale must be a normal FP32 value")
    return PreparedCausalAttention(
        backend,
        tokens,
        kv_tokens,
        query_heads,
        kv_heads,
        head_dim,
        dtype,
        backend.validate_device(dtype, device),
        float(scale),
    )


@dataclass(frozen=True)
class PreparedRoPE:
    """Rotate [T,H,D] using caller-owned FP32 tables [T,R/2].

    Tables must already select the desired positions and frequency convention.
    Only R channels rotate; remaining channels are copied. Exact x/out alias is
    permitted. FP32 opmath and one output cast; FP64 reference uses FP64 opmath.
    """

    _backend: object
    tokens: int
    heads: int
    head_dim: int
    rotary_dim: int
    layout: str
    dtype: torch.dtype
    device: torch.device
    workspace_bytes = 0

    def run(self, x, cos, sin, *, out):
        shape = (self.tokens, self.heads, self.head_dim)
        for name, tensor in (("x", x), ("out", out)):
            _tensor(tensor, shape, self.dtype, self.device, name)
        table_shape = (self.tokens, self.rotary_dim // 2)
        for name, tensor in (("cos", cos), ("sin", sin)):
            _tensor(tensor, table_shape, torch.float32, self.device, name)
        _no_overlap(out, (x,), allow_inplace=True)
        _no_overlap(out, (cos, sin))
        if self._backend.execution_kind == "torch":
            work = torch.float64 if self.dtype == torch.float64 else torch.float32
            data = x.to(work)
            half = self.rotary_dim // 2
            if self.layout == "split_half":
                a, b = data[..., :half], data[..., half : self.rotary_dim]
            else:
                a, b = (
                    data[..., : self.rotary_dim : 2],
                    data[..., 1 : self.rotary_dim : 2],
                )
            c, s = cos.to(work)[:, None, :], sin.to(work)[:, None, :]
            # Evaluate both results before writing, including exact in-place use.
            first, second = a * c - b * s, a * s + b * c
            if out.data_ptr() != x.data_ptr():
                out.copy_(x)
            if self.layout == "split_half":
                out[..., :half].copy_(first)
                out[..., half : self.rotary_dim].copy_(second)
            else:
                out[..., : self.rotary_dim : 2].copy_(first)
                out[..., 1 : self.rotary_dim : 2].copy_(second)
        else:
            with torch.cuda.device(self.device):
                self._backend._check(
                    self._backend._native.venturi_rope_run(
                        x.data_ptr(),
                        cos.data_ptr(),
                        sin.data_ptr(),
                        out.data_ptr(),
                        self.tokens,
                        self.heads,
                        self.head_dim,
                        self.rotary_dim,
                        int(self.layout == "interleaved"),
                        _DTYPES[self.dtype],
                        torch.cuda.current_stream(self.device).cuda_stream,
                    )
                )
        return out


@dataclass(frozen=True)
class PreparedCausalAttention:
    """Causal attention with a dense caller-owned prefix, without cache mutation.

    Q[T,Hq,D], K/V[S,Hkv,D]; Hq/Hkv adjacent query heads share each KV head.
    Query row i attends keys 0..S-T+i inclusive (bottom-right causal alignment).
    No bias, padding, sliding window or dropout. Inputs must yield finite FP32
    logits; overflow/non-finite inputs have no numerical guarantee.
    """

    _backend: object
    tokens: int
    kv_tokens: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    scale: float
    workspace_bytes = 0

    def run(self, q, k, v, *, out):
        qshape = (self.tokens, self.query_heads, self.head_dim)
        kvshape = (self.kv_tokens, self.kv_heads, self.head_dim)
        for name, tensor, shape in (
            ("q", q, qshape),
            ("k", k, kvshape),
            ("v", v, kvshape),
            ("out", out, qshape),
        ):
            _tensor(tensor, shape, self.dtype, self.device, name)
        _no_overlap(out, (q, k, v))
        if self._backend.execution_kind == "torch":
            work = torch.float64 if self.dtype == torch.float64 else torch.float32
            query = q.to(work).transpose(0, 1)
            repeats = self.query_heads // self.kv_heads
            key = k.to(work).repeat_interleave(repeats, dim=1).transpose(0, 1)
            value = v.to(work).repeat_interleave(repeats, dim=1).transpose(0, 1)
            scores = (query @ key.transpose(-1, -2)) * self.scale
            qi = torch.arange(self.tokens, device=self.device)[:, None]
            ki = torch.arange(self.kv_tokens, device=self.device)[None, :]
            scores.masked_fill_(ki > qi + self.kv_tokens - self.tokens, -torch.inf)
            out.copy_((scores.softmax(-1) @ value).transpose(0, 1).to(self.dtype))
        else:
            with torch.cuda.device(self.device):
                self._backend._check(
                    self._backend._native.venturi_attention_run(
                        q.data_ptr(),
                        k.data_ptr(),
                        v.data_ptr(),
                        out.data_ptr(),
                        self.tokens,
                        self.kv_tokens,
                        self.query_heads,
                        self.kv_heads,
                        self.head_dim,
                        _DTYPES[self.dtype],
                        self.scale,
                        torch.cuda.current_stream(self.device).cuda_stream,
                    )
                )
        return out
