"""Public prepared-operator contract, independent of a native library or model.

Capabilities are allocation-free *implementation* metadata, not hardware probes,
shape qualification, memory admission or whole-model support. Unknown operations,
quantization layouts and devices fail closed. There is no implicit CPU fallback.
The current tensor boundary is PyTorch; this does not implement Metal or MLX.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


class UnsupportedOperationError(ValueError):
    """The selected backend has no declared implementation for a request."""


@dataclass(frozen=True)
class OperatorSupport:
    operation: str
    device_types: frozenset[str]
    dtypes: frozenset[torch.dtype]
    weight_format: str | None = None
    weight_layout: str | None = None
    materializes_weights: bool = False

    def __post_init__(self):
        object.__setattr__(self, "device_types", frozenset(self.device_types))
        object.__setattr__(self, "dtypes", frozenset(self.dtypes))
        if (
            not isinstance(self.operation, str)
            or not self.operation
            or not self.device_types
            or not self.dtypes
        ):
            raise ValueError(
                "Operator support requires an operation, devices and dtypes"
            )
        if any(not isinstance(d, str) or not d for d in self.device_types):
            raise ValueError("Device types must be nonempty strings")
        if any(not isinstance(d, torch.dtype) for d in self.dtypes):
            raise TypeError("Capability dtypes must be torch.dtype values")
        if self.weight_layout is not None and self.weight_format is None:
            raise ValueError("A weight layout requires an explicit format")
        for value in (self.weight_format, self.weight_layout):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("Weight format/layout must be nonempty strings")
        if type(self.materializes_weights) is not bool:
            raise TypeError("Weight materialization flag must be bool")


@dataclass(frozen=True)
class BackendCapabilities:
    implementation: str
    operators: tuple[OperatorSupport, ...]
    reference: bool = False
    cuda_graphs: bool = False

    def __post_init__(self):
        object.__setattr__(self, "operators", tuple(self.operators))
        if not self.implementation or any(
            not isinstance(o, OperatorSupport) for o in self.operators
        ):
            raise ValueError(
                "Capabilities require an implementation and typed operators"
            )
        if type(self.reference) is not bool or type(self.cuda_graphs) is not bool:
            raise TypeError("Reference and graph capability flags must be bool")

    def supports(
        self, operation, *, dtype, device, weight_format=None, weight_layout=None
    ):
        # torch.device parses metadata only: do not initialize CUDA here.
        kind = torch.device(device).type
        if operation in ("linear", "embedding") and weight_format is None:
            weight_format = "dense"
        return any(
            o.operation == operation
            and kind in o.device_types
            and dtype in o.dtypes
            and weight_format == o.weight_format
            and weight_layout == o.weight_layout
            for o in self.operators
        )

    def require(self, operation, **request):
        if not self.supports(operation, **request):
            raise UnsupportedOperationError(
                f"{self.implementation} does not support {operation}: {request}; "
                "select an explicit backend/format, no automatic fallback"
            )


@runtime_checkable
class LinearPlan(Protocol):
    tokens: int
    dtype: torch.dtype
    device: torch.device
    in_features: int
    out_features: int
    workspace_bytes: int
    storage_tensors: tuple

    def run(self, x, *, out, workspace=None): ...


@runtime_checkable
class LinearWeight(Protocol):
    dtype: torch.dtype
    device: torch.device
    in_features: int
    out_features: int
    storage_tensors: tuple
    storage_nbytes: int

    def prepare(self, *, tokens) -> LinearPlan: ...

    def prepare_embedding(self, token_ids): ...


@runtime_checkable
class OperatorBackend(Protocol):
    """Structural interface; third-party backends need not inherit Backend.

    Upload owns storage; prepare binds shapes; run writes explicit outputs.
    Plans, operands and independent workspaces must outlive queued execution.
    Device validation, allocation and synchronization are setup operations, never
    hidden per-operator CPU/device transfers. Concrete backends own their plans.
    """

    name: str
    capabilities: BackendCapabilities

    def resolve_device(self, device) -> torch.device: ...
    def validate_device(self, dtype, device) -> torch.device: ...
    def require(self, operation, **request): ...
    def require_linear(self, weight, *, dtype, device): ...
    def empty(self, shape, *, dtype, device) -> torch.Tensor: ...
    def synchronize(self, device): ...
    def upload_linear(self, weight, *, dtype, device) -> LinearWeight: ...
    def upload_fp32_linear(self, weight, *, dtype, device) -> LinearWeight: ...
    def upload_rmsnorm(self, weight, *, dtype, device, epsilon): ...
    def prepare_linear(self, weight, *, tokens, dtype, device) -> LinearPlan: ...
    def prepare_rmsnorm(self, weight, *, tokens, dtype, device, epsilon): ...
    def prepare_swiglu(self, *, tokens, features, dtype, device): ...
    def prepare_add(self, *, tokens, features, dtype, device): ...
    def prepare_rope(self, **configuration): ...
    def prepare_attention(self, **configuration): ...


class Backend(ABC):
    """Optional base for the public contract, not an operator implementation.

    A new backend must supply its own upload and prepared-operator methods; it
    must not inherit CUDA internals or branch on a display name. State-space and
    model-specific fused operators are separate extensions to this core API.
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    def require(self, operation, **request):
        self.capabilities.require(operation, **request)

    def require_linear(self, weight, *, dtype, device):
        self.require(
            "linear",
            dtype=dtype,
            device=device,
            weight_format=weight.format,
            weight_layout=getattr(weight, "layout", None),
        )

    @abstractmethod
    def resolve_device(self, device) -> torch.device: ...

    @abstractmethod
    def validate_device(self, dtype, device) -> torch.device: ...

    @abstractmethod
    def empty(self, shape, *, dtype, device) -> torch.Tensor: ...

    @abstractmethod
    def synchronize(self, device): ...

    @abstractmethod
    def upload_linear(self, weight, *, dtype, device) -> LinearWeight: ...

    @abstractmethod
    def upload_fp32_linear(self, weight, *, dtype, device) -> LinearWeight: ...

    @abstractmethod
    def upload_rmsnorm(self, weight, *, dtype, device, epsilon): ...

    @abstractmethod
    def prepare_linear(self, weight, *, tokens, dtype, device) -> LinearPlan: ...

    @abstractmethod
    def prepare_rmsnorm(self, weight, *, tokens, dtype, device, epsilon): ...

    @abstractmethod
    def prepare_swiglu(self, *, tokens, features, dtype, device): ...

    @abstractmethod
    def prepare_add(self, *, tokens, features, dtype, device): ...

    @abstractmethod
    def prepare_rope(self, **configuration): ...

    @abstractmethod
    def prepare_attention(self, **configuration): ...


def create_backend(kind, *, library=None, q4_backend=None) -> OperatorBackend:
    """Explicit construction only; never build, probe devices or choose fallback.

    Native library loading happens only for an explicitly selected CUDA backend.
    Metal/MLX are intentionally not registered until implemented and qualified.
    """
    from .blocks import CudaBackend, TorchBackend

    if kind == "torch":
        if library is not None or q4_backend is not None:
            raise ValueError(
                "The torch reference backend takes no native library/plugins"
            )
        return TorchBackend()
    if kind == "cuda":
        if library is None:
            raise ValueError("CUDA requires an explicit prebuilt blocks library")
        return CudaBackend(library, q4_backend=q4_backend)
    raise UnsupportedOperationError(
        f"Backend {kind!r} is not implemented; select 'torch' or 'cuda' explicitly"
    )
