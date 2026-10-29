"""Reusable pre-normalized SwiGLU residual sublayer, not a whole model adapter.

Y = X + Down(cast(SiLU(Gate(RMSNorm(X))) * Up(RMSNorm(X)))).
Every operator output is cast to the declared activation dtype. No implicit
fusion, bias, attention, KV state or architecture detection is performed.
Qwen and Llama-style adapters may compose this same mathematical sublayer.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch

from ..ops.backend import OperatorBackend
from ..ops.blocks import _no_overlap, _shape, _tensor
from ..ops.weights import DenseWeight, Q4RowSplitWeight, Q8RowSplitWeight


@dataclass(frozen=True)
class GatedMLPWeights:
    norm: torch.Tensor
    gate: DenseWeight | Q4RowSplitWeight | Q8RowSplitWeight
    up: DenseWeight | Q4RowSplitWeight | Q8RowSplitWeight
    down: DenseWeight | Q4RowSplitWeight | Q8RowSplitWeight

    def __post_init__(self):
        for weight in (self.gate, self.up, self.down):
            if not isinstance(
                weight, (DenseWeight, Q4RowSplitWeight, Q8RowSplitWeight)
            ):
                raise TypeError(
                    "Gated MLP requires explicit dense, Q4 or Q8 weight descriptors"
                )
            weight.validate()
        h, intermediate = self.gate.in_features, self.gate.out_features
        if (self.up.in_features, self.up.out_features) != (h, intermediate):
            raise ValueError(
                "gate and up must have the same [intermediate, hidden] shape"
            )
        if (self.down.in_features, self.down.out_features) != (intermediate, h):
            raise ValueError("down must have shape [hidden, intermediate]")
        if (
            not isinstance(self.norm, torch.Tensor)
            or self.norm.shape != (h,)
            or not self.norm.is_floating_point()
        ):
            raise ValueError("norm must be a floating-point [hidden] vector")

    def validate_backend(self, backend: OperatorBackend, *, dtype, device):
        """Allocation-free core capability preflight, not a device/shape probe."""
        for operation in ("rmsnorm", "swiglu", "add"):
            backend.require(operation, dtype=dtype, device=device)
        for weight in (self.gate, self.up, self.down):
            backend.require_linear(weight, dtype=dtype, device=device)

    def upload(self, backend: OperatorBackend, *, dtype, device, epsilon=1e-6):
        """Own one weight copy. Bind decode/prefill shapes without another upload."""
        self.validate_backend(backend, dtype=dtype, device=device)
        return ResidentGatedMLP(
            backend,
            backend.upload_rmsnorm(
                self.norm, dtype=dtype, device=device, epsilon=epsilon
            ),
            backend.upload_linear(self.gate, dtype=dtype, device=device),
            backend.upload_linear(self.up, dtype=dtype, device=device),
            backend.upload_linear(self.down, dtype=dtype, device=device),
        )

    def prepare(self, backend, *, tokens, dtype, device, epsilon=1e-6):
        """Compatibility helper: upload then prepare one token shape."""
        _shape(tokens, self.gate.in_features)
        return self.upload(
            backend, dtype=dtype, device=device, epsilon=epsilon
        ).prepare(tokens=tokens)


@dataclass(frozen=True)
class ResidentGatedMLP:
    """Shared read-only weight storage; activation workspace stays per execution.

    Each shape has separate execution metadata/handles but the same tensor storage.
    This is not a KV cache, scheduler, or full-model adapter.
    """

    _backend: object
    norm: object
    gate: object
    up: object
    down: object

    @property
    def storage_tensors(self):
        return tuple(
            t
            for p in (self.norm, self.gate, self.up, self.down)
            for t in p.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)

    def prepare(self, *, tokens):
        _shape(tokens, self.gate.in_features)
        dtype, device = self.gate.dtype, self.gate.device
        # Prepare all stateless metadata before allocating any cuBLAS handles.
        swiglu = self._backend.prepare_swiglu(
            tokens=tokens, features=self.gate.out_features, dtype=dtype, device=device
        )
        add = self._backend.prepare_add(
            tokens=tokens, features=self.gate.in_features, dtype=dtype, device=device
        )
        return PreparedGatedMLP(
            self.norm.prepare(tokens=tokens),
            self.gate.prepare(tokens=tokens),
            self.up.prepare(tokens=tokens),
            self.down.prepare(tokens=tokens),
            swiglu,
            add,
            _empty=self._backend.empty,
        )


@dataclass(frozen=True)
class GatedMLPWorkspace:
    """Mutable activation/scratch buffers: never share across overlapping executions."""

    hidden: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    linear: torch.Tensor | None

    @property
    def nbytes(self):
        return sum(
            x.numel() * x.element_size()
            for x in (self.hidden, self.gate, self.up, self.linear)
            if x is not None
        )


@dataclass(frozen=True)
class PreparedGatedMLP:
    norm: object
    gate: object
    up: object
    down: object
    swiglu: object
    add: object
    # Preserve the original six-argument plan constructor. Normal preparation
    # injects the selected backend allocator; explicit legacy plans keep Torch.
    _empty: Callable = field(default=torch.empty, repr=False, compare=False)

    @property
    def dtype(self):
        return self.gate.dtype

    @property
    def device(self):
        return self.gate.device

    @property
    def tokens(self):
        return self.gate.tokens

    @property
    def hidden_size(self):
        return self.gate.in_features

    @property
    def intermediate_size(self):
        return self.gate.out_features

    @property
    def linear_workspace_bytes(self):
        return max(p.workspace_bytes for p in (self.gate, self.up, self.down))

    @property
    def storage_tensors(self):
        return tuple(
            t
            for p in (self.norm, self.gate, self.up, self.down)
            for t in p.storage_tensors
        )

    def empty_workspace(self):
        """Explicit allocation helper; use before timing/capture, once per execution lane."""
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Workspace allocation must run outside CUDA graph capture"
                    )
        options = dict(dtype=self.dtype, device=self.device)
        return GatedMLPWorkspace(
            self._empty((self.tokens, self.hidden_size), **options),
            self._empty((self.tokens, self.intermediate_size), **options),
            self._empty((self.tokens, self.intermediate_size), **options),
            self._empty(
                (self.linear_workspace_bytes,), dtype=torch.uint8, device=self.device
            )
            if self.linear_workspace_bytes
            else None,
        )

    def close(self):
        """Release execution handles only after queued work/graph replays complete.

        Shared weights are retained by other plans/residents; no implicit device
        synchronization is performed. Reference and Q4 plans have no handles.
        """
        for plan in (self.gate, self.up, self.down):
            close = getattr(plan, "close", None)
            if close is not None:
                close()

    def validate(self, x, *, out, workspace):
        """Metadata/alias preflight only; no kernels, state mutation or allocations."""
        shape = _shape(self.tokens, self.hidden_size)
        _tensor(x, shape, self.dtype, self.device, "x")
        _tensor(out, shape, self.dtype, self.device, "out")
        if not isinstance(workspace, GatedMLPWorkspace):
            raise TypeError("Gated MLP requires an explicit GatedMLPWorkspace")
        operands = [x, out]
        for name, tensor, expected in (
            ("hidden", workspace.hidden, shape),
            ("gate", workspace.gate, (self.tokens, self.intermediate_size)),
            ("up", workspace.up, (self.tokens, self.intermediate_size)),
        ):
            _tensor(tensor, expected, self.dtype, self.device, "workspace." + name)
            _no_overlap(tensor, operands)
            operands.append(tensor)
        if self.linear_workspace_bytes:
            _tensor(
                workspace.linear,
                (self.linear_workspace_bytes,),
                torch.uint8,
                self.device,
                "workspace.linear",
            )
            _no_overlap(workspace.linear, operands)
            if workspace.linear.data_ptr() % 256:
                raise ValueError("cuBLAS workspace requires 256-byte alignment")
            operands.append(workspace.linear)
        elif workspace.linear is not None:
            raise ValueError("No linear workspace is required by this plan")
        _no_overlap(out, (x,))
        for mutable in operands[1:]:
            _no_overlap(mutable, self.storage_tensors)

    def run(self, x, *, out, workspace):
        self.validate(x, out=out, workspace=workspace)
        self.norm.run(x, out=workspace.hidden)
        self.gate.run(
            workspace.hidden,
            out=workspace.gate,
            workspace=workspace.linear if self.gate.workspace_bytes else None,
        )
        self.up.run(
            workspace.hidden,
            out=workspace.up,
            workspace=workspace.linear if self.up.workspace_bytes else None,
        )
        self.swiglu.run(workspace.gate, workspace.up, out=workspace.gate)
        # Normalized activations are dead; reuse their buffer for down projection.
        self.down.run(
            workspace.gate,
            out=workspace.hidden,
            workspace=workspace.linear if self.down.workspace_bytes else None,
        )
        self.add.run(x, workspace.hidden, out=out)
        return out
