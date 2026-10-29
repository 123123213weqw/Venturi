"""Packed query/gate QK normalization + partial split-half RoPE, and output gate.

Observable activation rounding boundaries follow Qwen's eager norm/rotary/gate
contract. mRoPE axis interleaving is a table convention, NOT even/odd pairing.
No cache manager, position generation, attention, projection or implicit copies.
"""

from dataclasses import dataclass

import torch

from .blocks import _DTYPES, _epsilon, _no_overlap, _shape, _tensor
from .state_space import _owned, _ready


def upload_qwen_qk_norm(
    backend, query_weight, key_weight, *, dtype, device, epsilon=1e-6
):
    target = backend.validate_device(dtype, device)
    epsilon = _epsilon(epsilon)
    if (
        not isinstance(query_weight, torch.Tensor)
        or query_weight.ndim != 1
        or query_weight.numel() not in (64, 128, 256)
    ):
        raise ValueError("QK norm requires head_dim 64, 128 or 256")
    work = torch.float64 if dtype == torch.float64 else torch.float32
    gamma = tuple(
        1
        + _owned(
            backend, w, dtype=work, device=target, shape=query_weight.shape, name=name
        )
        for w, name in ((query_weight, "query norm"), (key_weight, "key norm"))
    )
    if not all(bool(torch.isfinite(w).all()) for w in gamma):
        raise ValueError("Effective zero-centered QK gamma must be finite")
    _ready(target)
    return ResidentQwenQKNorm(backend, *gamma, dtype, epsilon)


@dataclass(frozen=True)
class ResidentQwenQKNorm:
    backend: object
    query_gamma: torch.Tensor
    key_gamma: torch.Tensor
    dtype: torch.dtype
    epsilon: float

    @property
    def device(self):
        return self.query_gamma.device

    @property
    def head_dim(self):
        return self.query_gamma.numel()

    @property
    def storage_tensors(self):
        return (self.query_gamma, self.key_gamma)

    def prepare(self, *, tokens, query_heads, kv_heads, rotary_dim):
        _shape(tokens, query_heads)
        _shape(kv_heads, rotary_dim)
        if (
            query_heads % kv_heads
            or rotary_dim > self.head_dim
            or rotary_dim % 2
            or tokens * (query_heads + kv_heads) >= 2**31
            or 2 * query_heads * self.head_dim >= 2**31
        ):
            raise ValueError("Unsupported QK head/rotary geometry")
        self.backend.validate_device(self.dtype, self.device)
        return PreparedQwenQKNormRoPE(self, tokens, query_heads, kv_heads, rotary_dim)


@dataclass(frozen=True)
class PreparedQwenQKNormRoPE:
    resident: ResidentQwenQKNorm
    tokens: int
    query_heads: int
    kv_heads: int
    rotary_dim: int

    def validate(self, qg, key, cos, sin, *, query_out, key_out, norm_only=False):
        r = self.resident
        d = r.head_dim
        operands = (
            ("qg", qg, (self.tokens, 2 * self.query_heads * d), r.dtype),
            ("key", key, (self.tokens, self.kv_heads, d), r.dtype),
            ("query_out", query_out, (self.tokens, self.query_heads, d), r.dtype),
            ("key_out", key_out, (self.tokens, self.kv_heads, d), r.dtype),
            ("cos", cos, (self.tokens, self.rotary_dim // 2), torch.float32),
            ("sin", sin, (self.tokens, self.rotary_dim // 2), torch.float32),
        )
        for name, tensor, shape, dtype in operands:
            if norm_only and name in ("cos", "sin"):
                continue
            _tensor(tensor, shape, dtype, r.device, name)
        tables = () if norm_only else (cos, sin)
        _no_overlap(query_out, (qg, key, key_out) + tables + r.storage_tensors)
        _no_overlap(key_out, (qg, key) + tables + r.storage_tensors)

    def run_normalized(self, qg, key, *, query_out, key_out):
        """Expose the same rounded norm stage, without RoPE, for composition/qualification."""
        self.validate(
            qg, key, None, None, query_out=query_out, key_out=key_out, norm_only=True
        )
        r = self.resident
        if r.backend.execution_kind == "torch":
            work = torch.float64 if r.dtype == torch.float64 else torch.float32
            q = qg.reshape(self.tokens, self.query_heads, 2 * r.head_dim)[
                ..., : r.head_dim
            ]
            for raw, gamma, out in (
                (q, r.query_gamma, query_out),
                (key, r.key_gamma, key_out),
            ):
                value = raw.to(work)
                out.copy_(
                    (
                        value
                        * torch.rsqrt(value.square().mean(-1, keepdim=True) + r.epsilon)
                        * gamma
                    ).to(r.dtype)
                )
        else:
            with torch.cuda.device(r.device):
                r.backend._check(
                    r.backend._native.venturi_qwen_qk_run(
                        *(t.data_ptr() for t in (qg, key, r.query_gamma, r.key_gamma)),
                        0,
                        0,
                        query_out.data_ptr(),
                        key_out.data_ptr(),
                        self.tokens,
                        self.query_heads,
                        self.kv_heads,
                        r.head_dim,
                        0,
                        _DTYPES[r.dtype],
                        r.epsilon,
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return query_out, key_out

    def run(self, qg, key, cos, sin, *, query_out, key_out):
        """Q/G are interleaved PER HEAD. Norm rounds before staged rotary products.

        FP32 tables are rounded to the activation dtype; each rotation product
        rounds before the sum. All outputs are distinct, with no implicit unpack.
        """
        self.validate(qg, key, cos, sin, query_out=query_out, key_out=key_out)
        r = self.resident
        d = r.head_dim
        if r.backend.execution_kind == "torch":
            work = torch.float64 if r.dtype == torch.float64 else torch.float32
            half = self.rotary_dim // 2
            c, s = (t.to(r.dtype).to(work)[:, None, :] for t in (cos, sin))
            for raw, gamma, out in (
                (
                    qg.reshape(self.tokens, self.query_heads, 2 * d)[..., :d],
                    r.query_gamma,
                    query_out,
                ),
                (key, r.key_gamma, key_out),
            ):
                values = raw.to(work)
                norm = (
                    values
                    * torch.rsqrt(values.square().mean(-1, keepdim=True) + r.epsilon)
                    * gamma
                ).to(r.dtype)
                first, second = (
                    norm[..., :half].to(work),
                    norm[..., half : self.rotary_dim].to(work),
                )
                left = (
                    (first * c).to(r.dtype).to(work) - (second * s).to(r.dtype).to(work)
                ).to(r.dtype)
                right = (
                    (second * c).to(r.dtype).to(work) + (first * s).to(r.dtype).to(work)
                ).to(r.dtype)
                out.copy_(norm)
                out[..., :half].copy_(left)
                out[..., half : self.rotary_dim].copy_(right)
        else:
            with torch.cuda.device(r.device):
                r.backend._check(
                    r.backend._native.venturi_qwen_qk_run(
                        *(
                            t.data_ptr()
                            for t in (
                                qg,
                                key,
                                r.query_gamma,
                                r.key_gamma,
                                cos,
                                sin,
                                query_out,
                                key_out,
                            )
                        ),
                        self.tokens,
                        self.query_heads,
                        self.kv_heads,
                        d,
                        self.rotary_dim,
                        _DTYPES[r.dtype],
                        r.epsilon,
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return query_out, key_out


def prepare_qwen_output_gate(backend, *, tokens, heads, head_dim, dtype, device):
    _shape(tokens, heads)
    _shape(head_dim, 1)
    if head_dim not in (64, 128, 256) or 2 * heads * head_dim >= 2**31:
        raise ValueError("Unsupported gated attention head geometry")
    return PreparedQwenOutputGate(
        backend, tokens, heads, head_dim, dtype, backend.validate_device(dtype, device)
    )


@dataclass(frozen=True)
class PreparedQwenOutputGate:
    backend: object
    tokens: int
    heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def validate(self, x, qg, *, out):
        for name, tensor, shape in (
            ("attention", x, (self.tokens, self.heads, self.head_dim)),
            ("out", out, (self.tokens, self.heads, self.head_dim)),
            ("qg", qg, (self.tokens, 2 * self.heads * self.head_dim)),
        ):
            _tensor(tensor, shape, self.dtype, self.device, name)
        _no_overlap(out, (x,), allow_inplace=True)
        _no_overlap(out, (qg,))

    def run(self, x, qg, *, out):
        """Read packed gate without a copy; sigmoid rounds before multiplication."""
        self.validate(x, qg, out=out)
        if self.backend.execution_kind == "torch":
            work = torch.float64 if self.dtype == torch.float64 else torch.float32
            gate = qg.reshape(self.tokens, self.heads, 2 * self.head_dim)[
                ..., self.head_dim :
            ]
            sigmoid = gate.to(work).sigmoid().to(self.dtype).to(work)
            out.copy_((x.to(work) * sigmoid).to(self.dtype))
        else:
            with torch.cuda.device(self.device):
                self.backend._check(
                    self.backend._native.venturi_qwen_gate_run(
                        *(t.data_ptr() for t in (x, qg, out)),
                        self.tokens,
                        self.heads,
                        self.head_dim,
                        _DTYPES[self.dtype],
                        torch.cuda.current_stream(self.device).cuda_stream,
                    )
                )
        return out
