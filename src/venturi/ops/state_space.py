"""Native causal convolution, GDN gate transforms and staged gated RMSNorm.

These are reusable operators, not a hybrid Qwen forward or state scheduler.
Convolution uses chronological [K-1,C] history and [K,C] checkpoint weights.
Explicit activation rounding boundaries follow the documented mathematical
contract; preparation may transform constants, execution never falls back.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .blocks import _DTYPES, _epsilon, _no_overlap, _shape, _tensor


def _owned(backend, weight, *, dtype, device, shape, name):
    if (
        not isinstance(weight, torch.Tensor)
        or weight.shape != shape
        or not weight.is_floating_point()
    ):
        raise ValueError(f"{name} requires a floating-point tensor of shape {shape}")
    owned = weight.detach().to(dtype=dtype, device=device, copy=True).contiguous()
    if not bool(torch.isfinite(owned).all()):
        raise ValueError(f"{name} must be finite in its resident dtype")
    return owned


def _ready(device):
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def upload_causal_conv(backend, weight, *, dtype, device):
    target = backend.validate_device(dtype, device)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("Causal convolution weight requires [kernel,channels]")
    kernel, channels = weight.shape
    _shape(kernel, channels)
    if kernel > 32:
        raise ValueError("Causal convolution currently supports kernel widths 1..32")
    if weight.dtype != dtype and not (
        weight.dtype == torch.float32 and dtype in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "Convolution needs matching dtype or explicit FP32 parameter storage"
        )
    owned = _owned(
        backend,
        weight,
        dtype=weight.dtype,
        device=target,
        shape=(kernel, channels),
        name="conv weight",
    )
    _ready(target)
    return ResidentCausalConv(backend, owned, dtype)


@dataclass(frozen=True)
class ResidentCausalConv:
    backend: object
    _weight: torch.Tensor
    dtype: torch.dtype

    @property
    def storage_tensors(self):
        return (self._weight,)

    def prepare(self, *, tokens):
        _shape(tokens, self._weight.shape[1])
        self.backend.validate_device(self.dtype, self._weight.device)
        return PreparedCausalConv(self, tokens)


@dataclass(frozen=True)
class PreparedCausalConv:
    resident: ResidentCausalConv
    tokens: int

    def run(self, x, *, history, out, history_out):
        """SiLU(cast(conv(x))); exact history/history_out alias is safe.

        history contains raw projected inputs, oldest first, not activated
        outputs. Zero it for a new request. No hidden reset/commit is performed;
        the caller must serialize mutation and discard state on failure.
        """
        r = self.resident
        kernel, channels = r._weight.shape
        dtype, device = r.dtype, r._weight.device
        for name, tensor, shape in (
            ("x", x, (self.tokens, channels)),
            ("out", out, (self.tokens, channels)),
            ("history", history, (kernel - 1, channels)),
            ("history_out", history_out, (kernel - 1, channels)),
        ):
            _tensor(tensor, shape, dtype, device, name)
        _no_overlap(out, (x, history, history_out, r._weight))
        _no_overlap(history_out, (x, r._weight))
        _no_overlap(history_out, (history,), allow_inplace=True)
        if r.backend.execution_kind == "torch":
            work = torch.float64 if dtype == torch.float64 else torch.float32
            joined = torch.cat((history, x))
            acc = torch.zeros_like(x, dtype=work)
            for tap in range(kernel):
                acc.add_(
                    joined[tap : tap + self.tokens].to(work) * r._weight[tap].to(work)
                )
            # Explicit conv-output rounding before activation, unlike a one-cast
            # fused variant. Compare kernels using this same staged contract.
            conv = acc.to(dtype).to(work)
            out.copy_(F.silu(conv).to(dtype))
            if kernel > 1:
                history_out.copy_(joined[-(kernel - 1) :])
        else:
            with torch.cuda.device(device):
                r.backend._check(
                    r.backend._native.venturi_conv_run(
                        *[
                            t.data_ptr()
                            for t in (x, r._weight, history, out, history_out)
                        ],
                        self.tokens,
                        channels,
                        kernel,
                        _DTYPES[dtype],
                        int(r._weight.dtype == torch.float32),
                        torch.cuda.current_stream(device).cuda_stream,
                    )
                )
        return out


def upload_gdn_gates(backend, a_log, dt_bias, *, dtype, device):
    target = backend.validate_device(dtype, device)
    if not isinstance(a_log, torch.Tensor) or a_log.ndim != 1:
        raise ValueError("A_log must be a floating-point [heads] vector")
    _shape(1, a_log.numel())
    work = torch.float64 if dtype == torch.float64 else torch.float32
    log = _owned(
        backend, a_log, dtype=work, device=target, shape=a_log.shape, name="A_log"
    )
    bias = _owned(
        backend, dt_bias, dtype=work, device=target, shape=a_log.shape, name="dt_bias"
    )
    # Transform a constant once, never run PyTorch exp during native inference.
    rate = log.exp()
    if not bool(torch.isfinite(rate).all()):
        raise ValueError("exp(A_log) must be finite in the gate arithmetic dtype")
    _ready(target)
    return ResidentGDNGates(backend, rate, bias, dtype)


@dataclass(frozen=True)
class ResidentGDNGates:
    backend: object
    _rate: torch.Tensor
    _bias: torch.Tensor
    dtype: torch.dtype

    @property
    def storage_tensors(self):
        return (self._rate, self._bias)

    def prepare(self, *, tokens, round_beta=True):
        _shape(tokens, self._rate.numel())
        if type(round_beta) is not bool:
            raise ValueError("round_beta must be boolean")
        self.backend.validate_device(self.dtype, self._rate.device)
        return PreparedGDNGates(self, tokens, round_beta)


@dataclass(frozen=True)
class PreparedGDNGates:
    resident: ResidentGDNGates
    tokens: int
    round_beta: bool

    def run(self, a, b, *, log_decay, beta):
        """Return -exp(A_log)*softplus(a+dt_bias), sigmoid(b) in FP32.

        With round_beta=True, sigmoid(b) first rounds to activation precision,
        matching an activation-typed sigmoid. False explicitly selects FP32
        sigmoid semantics. Finite inputs/results are required, not scanned here.
        """
        r = self.resident
        shape, device = (self.tokens, r._rate.numel()), r._rate.device
        for name, tensor in (("a", a), ("b", b)):
            _tensor(tensor, shape, r.dtype, device, name)
        for name, tensor in (("log_decay", log_decay), ("beta", beta)):
            _tensor(tensor, shape, r._rate.dtype, device, name)
        _no_overlap(log_decay, (a, b, beta) + r.storage_tensors)
        _no_overlap(beta, (a, b) + r.storage_tensors)
        if r.backend.execution_kind == "torch":
            work = r._rate.dtype
            log_decay.copy_(-r._rate * F.softplus(a.to(work) + r._bias))
            value = b.to(work).sigmoid()
            if self.round_beta:
                value = value.to(r.dtype).to(work)
            beta.copy_(value)
        else:
            with torch.cuda.device(device):
                r.backend._check(
                    r.backend._native.venturi_gates_run(
                        *[
                            t.data_ptr()
                            for t in (a, b, r._rate, r._bias, log_decay, beta)
                        ],
                        self.tokens,
                        shape[1],
                        _DTYPES[r.dtype],
                        int(self.round_beta),
                        torch.cuda.current_stream(device).cuda_stream,
                    )
                )
        return log_decay, beta


def upload_gated_rmsnorm(backend, weight, *, dtype, device, epsilon=1e-6):
    target = backend.validate_device(dtype, device)
    epsilon = _epsilon(epsilon)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
        raise ValueError("Gated RMSNorm weight requires a [features] vector")
    _shape(1, weight.numel())
    if weight.dtype not in (dtype, torch.float32):
        raise ValueError("Gated RMSNorm requires activation-dtype or FP32 weights")
    owned = _owned(
        backend,
        weight,
        dtype=torch.float64 if dtype == torch.float64 else weight.dtype,
        device=target,
        shape=weight.shape,
        name="gated norm weight",
    )
    _ready(target)
    return ResidentGatedRMSNorm(
        backend, owned, epsilon, dtype, weight.dtype != torch.float32
    )


@dataclass(frozen=True)
class ResidentGatedRMSNorm:
    backend: object
    _weight: torch.Tensor
    epsilon: float
    dtype: torch.dtype
    round_weighted: bool

    @property
    def storage_tensors(self):
        return (self._weight,)

    def prepare(self, *, tokens, heads):
        _shape(tokens, heads)
        if tokens * heads >= 2**31:
            raise ValueError("Gated RMSNorm row count must fit int32")
        self.backend.validate_device(self.dtype, self._weight.device)
        return PreparedGatedRMSNorm(self, tokens, heads)


@dataclass(frozen=True)
class PreparedGatedRMSNorm:
    resident: ResidentGatedRMSNorm
    tokens: int
    heads: int

    def run(self, x, z, *, out):
        """Normalize/cast, multiply gamma with its declared precision, gate/cast.

        Activation-dtype gamma retains its weighted activation cast. FP32 gamma
        instead promotes the product and does NOT add that cast (the original
        Qwen checkpoint convention). Gamma is shared across heads, not zero-centered.
        """
        r = self.resident
        shape = (self.tokens, self.heads, r._weight.numel())
        dtype, device = r.dtype, r._weight.device
        for name, tensor in (("x", x), ("z", z), ("out", out)):
            _tensor(tensor, shape, dtype, device, name)
        _no_overlap(out, (x, z, r._weight))
        if r.backend.execution_kind == "torch":
            work = torch.float64 if dtype == torch.float64 else torch.float32
            value = x.to(work)
            normal = (
                value * torch.rsqrt(value.square().mean(-1, keepdim=True) + r.epsilon)
            ).to(dtype)
            weighted = normal.to(work) * r._weight.to(work)
            if r.round_weighted:
                weighted = weighted.to(dtype)
            out.copy_((weighted.to(work) * F.silu(z.to(work))).to(dtype))
        else:
            with torch.cuda.device(device):
                r.backend._check(
                    r.backend._native.venturi_gated_rms_run(
                        *[t.data_ptr() for t in (x, z, r._weight, out)],
                        self.tokens,
                        self.heads,
                        shape[2],
                        _DTYPES[dtype],
                        int(not r.round_weighted),
                        r.epsilon,
                        torch.cuda.current_stream(device).cuda_stream,
                    )
                )
        return out


def upload_zero_centered_rmsnorm(backend, weight, *, dtype, device, epsilon=1e-6):
    """Prepare gamma=1+checkpoint_weight in FP32 (FP64 diagnostic mode).

    Reuses the existing norm kernel. No per-forward add or hidden numerical
    fallback. The source checkpoint tensor remains unchanged.
    """
    backend.validate_device(dtype, device)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 1
        or not weight.is_floating_point()
    ):
        raise ValueError("Zero-centered norm weight requires a floating-point vector")
    work = torch.float64 if dtype == torch.float64 else torch.float32
    return backend.upload_rmsnorm(
        1 + weight.detach().to(work), dtype=dtype, device=device, epsilon=epsilon
    )
