"""Explicitly prepared Ada backend, usable without an engine or model adapter."""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch

from .weights import Q4RowSplitWeight


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    # All accepted operands are contiguous, so their live byte intervals suffice.
    return a.device == b.device and max(a.data_ptr(), b.data_ptr()) < min(
        a.data_ptr() + a.numel() * a.element_size(),
        b.data_ptr() + b.numel() * b.element_size(),
    )


class AdaBackend:
    """Optional native plugin. Loading does not compile code or allocate GPU weights."""

    name = "venturi-ada"
    abi_version = 1

    def __init__(self, library: str | Path):
        self.library = Path(library).expanduser().resolve(strict=True)
        self._native = ctypes.CDLL(str(self.library))
        self._native.venturi_ada_abi.argtypes = []
        self._native.venturi_ada_abi.restype = ctypes.c_int
        self._native.venturi_ada_error.argtypes = []
        self._native.venturi_ada_error.restype = ctypes.c_char_p
        self._native.venturi_q4_prepare.argtypes = [ctypes.c_int] * 5
        self._native.venturi_q4_prepare.restype = ctypes.c_int
        self._native.venturi_q4_run.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        )
        self._native.venturi_q4_run.restype = ctypes.c_int
        if self._native.venturi_ada_abi() != self.abi_version:
            raise RuntimeError("Venturi Ada native ABI mismatch; rebuild the backend")

    def _error(self) -> RuntimeError:
        return RuntimeError(self._native.venturi_ada_error().decode("utf-8", "replace"))

    def prepare_linear(
        self,
        weight: Q4RowSplitWeight,
        *,
        tokens: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda:0",
        route: str = "auto",
    ) -> "PreparedLinear":
        """Own GPU copies and initialize the fixed-shape route outside graph capture.

        Preparation may allocate, copy, and synchronize. No model architecture is
        consulted. A separate plan is needed for each token count and dtype.
        """
        if not isinstance(weight, Q4RowSplitWeight):
            raise TypeError("Venturi Ada currently supports Q4RowSplitWeight only")
        weight.validate()
        if type(tokens) is not int or not 0 < tokens <= 4 * 65535:
            raise ValueError("tokens must be an integer in [1, 262140]")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("activations and output must be float16 or bfloat16")
        if route not in ("auto", "generic", "tuned", "mma"):
            raise ValueError("route must be auto, generic, tuned, or mma")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("Ada execution requires an explicit CUDA device")
        with torch.cuda.device(target):
            target = torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare_linear must run outside CUDA graph capture")
            if torch.cuda.get_device_capability(target) != (8, 9):
                raise ValueError(
                    "Venturi Ada currently requires compute capability 8.9"
                )
            native_route = self._native.venturi_q4_prepare(
                weight.out_features,
                weight.in_features,
                tokens,
                int(dtype == torch.bfloat16),
                {"auto": -1, "generic": 0, "tuned": 1, "mma": 2}[route],
            )
            if native_route < 0:
                raise self._error()
            # Explicit copies also detach preparation from any autograd graph.
            codes = weight.codes.detach().to(target, copy=True)
            scales = weight.scales.detach().to(target, copy=True)
            if not bool(torch.isfinite(scales).all()):
                raise ValueError("scales must be finite")
            owned = Q4RowSplitWeight(codes, scales, weight.in_features)
            # Establish ownership readiness before execution on any caller stream.
            torch.cuda.current_stream(target).synchronize()
        return PreparedLinear(self, owned, tokens, dtype, native_route)


@dataclass(frozen=True)
class PreparedLinear:
    """Fixed shape Linear: out[T,N] = x[T,K] @ W[N,K].T, without bias.

    run writes caller-owned output and performs no device allocation, full-weight
    decode, implicit cast, synchronization, or workspace construction. Keep this
    plan and graph operands alive until all asynchronous work/replays complete.
    """

    workspace_bytes = 0

    _backend: AdaBackend
    _weight: Q4RowSplitWeight
    tokens: int
    dtype: torch.dtype
    _native_route: int

    @property
    def device(self):
        return self._weight.codes.device

    @property
    def in_features(self):
        return self._weight.in_features

    @property
    def out_features(self):
        return self._weight.out_features

    @property
    def route(self):
        return {0: "generic-simt", 1: "ada-mma-small-t", 2: "ada-mma-general"}[
            self._native_route
        ]

    def empty_output(self) -> torch.Tensor:
        """Explicit allocation helper; call before timing or graph capture."""
        return torch.empty(
            (self.tokens, self.out_features), dtype=self.dtype, device=self.device
        )

    def run(self, x: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and (x.requires_grad or out.requires_grad):
            raise ValueError("Venturi Ada is inference-only; disable gradients")
        self._weight.validate()
        for name, tensor, shape in (
            ("x", x, (self.tokens, self.in_features)),
            ("out", out, (self.tokens, self.out_features)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != self.dtype
                or tensor.device != self.device
            ):
                raise ValueError(
                    f"{name} must have shape {shape}, dtype {self.dtype}, device {self.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous; no implicit copy is performed"
                )
            alignment = 16 if self._native_route in (1, 2) and name == "x" else 2
            if tensor.data_ptr() % alignment:
                raise ValueError(f"{name} requires {alignment}-byte alignment")
        for operand in (x, self._weight.codes, self._weight.scales):
            if _overlaps(out, operand):
                raise ValueError("output must not alias input or weight storage")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            result = self._backend._native.venturi_q4_run(
                x.data_ptr(),
                self._weight.codes.data_ptr(),
                self._weight.scales.data_ptr(),
                out.data_ptr(),
                self.out_features,
                self.in_features,
                self._weight.padded_k,
                self.tokens,
                int(self.dtype == torch.bfloat16),
                self._native_route,
                stream.cuda_stream,
            )
            if result < 0:
                raise self._backend._error()
        return out
