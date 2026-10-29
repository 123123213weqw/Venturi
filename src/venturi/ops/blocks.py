"""Prepared, model-independent operators with explicit buffers and stream ownership.

The native backend owns RMSNorm/SwiGLU/Add/RoPE/attention kernels and delegates dense GEMM to
cuBLAS, not NInfer. An explicitly supplied AdaBackend provides the existing
attributed Q4 operator. TorchBackend is an allocating numerical/performance
reference, not a hidden CUDA fallback. Plans and operands must outlive queued
GPU work; each concurrent execution needs independent caller-owned workspace.
"""

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import weakref

import torch
from torch.nn import functional as F

from .ada import _overlaps
from .backend import Backend, BackendCapabilities, OperatorSupport
from .weights import DenseWeight, Q4RowSplitWeight, Q8RowSplitWeight


_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


def _shape(tokens, features):
    if any(type(v) is not int or not 0 < v < 2**31 for v in (tokens, features)):
        raise ValueError("tokens and features must be positive int32 values")
    return (tokens, features)


def _tensor(tensor, shape, dtype, device, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"{name} requires shape {shape}, dtype {dtype}, device {device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous; no implicit copy")
    if torch.is_grad_enabled() and tensor.requires_grad:
        raise ValueError("Prepared operators are inference-only")


def _no_overlap(out, operands, *, allow_inplace=False):
    for operand in operands:
        if _overlaps(out, operand) and not (
            allow_inplace
            and out.data_ptr() == operand.data_ptr()
            and out.shape == operand.shape
            and out.dtype == operand.dtype
        ):
            raise ValueError("output/workspace must not alias operand storage")


def _epsilon(value):
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or not torch.finfo(torch.float32).tiny
        <= value
        <= torch.finfo(torch.float32).max
        or ctypes.c_float(value).value == 0
    ):
        raise ValueError("epsilon must be finite, positive, and a normal FP32 value")
    return float(value)


@dataclass(frozen=True)
class ResidentLinear:
    """One owned weight upload, many shape plans; storage is read-only by contract.

    Tensor views are exposed for diagnostics, not mutation. Plans retain storage;
    callers must retain plans through all asynchronous executions/graph replays.
    """

    _backend: object
    _payload: object
    storage_tensors: tuple
    dtype: torch.dtype
    device: torch.device
    in_features: int
    out_features: int

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)

    def prepare_embedding(self, token_ids):
        from .embedding import prepare_embedding
        from .vocabulary import ResidentQ8Vocabulary

        if isinstance(self._payload, ResidentQ8Vocabulary):
            return self._payload.prepare_embedding(token_ids)
        return prepare_embedding(self, token_ids)

    def prepare(self, *, tokens):
        _shape(tokens, self.in_features)
        self._backend.validate_device(self.dtype, self.device)
        return self._backend.bind_linear(self, tokens=tokens)


def _upload_fp32_linear(backend, weight, *, dtype, device):
    if not isinstance(weight, DenseWeight) or weight.data.dtype != torch.float32:
        raise ValueError(
            "Explicit FP32 Linear requires a DenseWeight with FP32 storage"
        )
    weight.validate()
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "FP32-parameter Linear requires explicit FP16/BF16 activations"
        )
    target = backend.validate_device(dtype, device)
    owned = weight.data.detach().to(target, copy=True)
    if not bool(torch.isfinite(owned).all()):
        raise ValueError("FP32 Linear weights must be finite")
    if target.type == "cuda":
        torch.cuda.current_stream(target).synchronize()
    payload = (owned, False) if backend.execution_kind == "torch" else owned
    return ResidentLinear(
        backend,
        payload,
        (owned,),
        dtype,
        target,
        weight.in_features,
        weight.out_features,
    )


@dataclass(frozen=True)
class ResidentRMSNorm:
    """One owned norm vector, shared across token-shape plans."""

    _backend: object
    _weight: torch.Tensor
    dtype: torch.dtype
    epsilon: float

    @property
    def storage_tensors(self):
        return (self._weight,)

    @property
    def storage_nbytes(self):
        return self._weight.numel() * self._weight.element_size()

    @property
    def device(self):
        return self._weight.device

    @property
    def features(self):
        return self._weight.numel()

    def prepare(self, *, tokens):
        _shape(tokens, self.features)
        self._backend.validate_device(self.dtype, self.device)
        return self._backend.bind_rmsnorm(self, tokens=tokens)


def _block_capabilities(*, reference, architecture=None, q4=False):
    devices = frozenset(("cpu", "cuda") if reference else ("cuda",))
    dtypes = frozenset(
        (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        if reference
        else (
            (torch.float16, torch.float32)
            if architecture == 70
            else (torch.float16, torch.bfloat16, torch.float32)
        )
    )
    operators = [
        OperatorSupport(op, devices, dtypes)
        for op in ("rmsnorm", "swiglu", "add", "rope", "attention")
    ]
    operators.append(OperatorSupport("linear", devices, dtypes, "dense"))
    operators.append(OperatorSupport("embedding", devices, dtypes, "dense"))
    quant_dtypes = dtypes if reference else dtypes & {torch.float16, torch.bfloat16}
    operators.append(
        OperatorSupport(
            "linear",
            devices,
            quant_dtypes,
            "q8_g32_fp16",
            "row_split_k128_v1",
            materializes_weights=reference,
        )
    )
    operators.append(
        OperatorSupport(
            "embedding",
            devices,
            quant_dtypes,
            "q8_g32_fp16",
            "row_split_k128_v1",
            materializes_weights=reference,
        )
    )
    if reference or (q4 and architecture == 89):
        operators.append(
            OperatorSupport(
                "linear",
                devices,
                quant_dtypes,
                "q4_g64_fp16",
                "row_split_k128_v1",
                materializes_weights=reference,
            )
        )
    return BackendCapabilities(
        "torch-reference" if reference else f"venturi-blocks-sm{architecture}",
        tuple(operators),
        reference=reference,
        cuda_graphs=not reference,
    )


class _SharedPreparedBackend(Backend):
    """Existing Torch/CUDA adapters only; new backends implement Backend directly.

    Common setup and extension entry points live here, not in model adapters.
    Legacy private aliases remain for callers outside this module.
    """

    def _device(self, dtype, device):
        return self.validate_device(dtype, device)

    def _bind_linear(self, resident, tokens):
        return self.bind_linear(resident, tokens=tokens)

    def _bind_rmsnorm(self, resident, tokens):
        return self.bind_rmsnorm(resident, tokens=tokens)

    def empty(self, shape, *, dtype, device):
        if not isinstance(shape, (list, tuple)) or any(
            type(d) is not int or not 0 <= d < 2**31 for d in shape
        ):
            raise ValueError("Allocation shape must contain nonnegative int32 extents")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("Allocation requires an explicit torch.dtype")
        if dtype.is_complex:
            raise ValueError("Backend buffers do not support complex dtypes")
        target = (
            self.validate_device(dtype, device)
            if dtype.is_floating_point
            else self.resolve_device(device)
        )
        return torch.empty(tuple(shape), dtype=dtype, device=target)

    def synchronize(self, device):
        target = self.resolve_device(device)
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def upload_fp32_linear(self, weight, *, dtype, device):
        """Explicit FP32 parameter / FP16 or BF16 activation path, no weight cast."""
        return _upload_fp32_linear(self, weight, dtype=dtype, device=device)

    def prepare_linear(self, weight, *, tokens, dtype, device):
        _shape(tokens, 1)
        return self.upload_linear(weight, dtype=dtype, device=device).prepare(
            tokens=tokens
        )

    def prepare_rmsnorm(self, weight, *, tokens, dtype, device, epsilon):
        _shape(tokens, 1)
        return self.upload_rmsnorm(
            weight, dtype=dtype, device=device, epsilon=epsilon
        ).prepare(tokens=tokens)

    def upload_causal_conv(self, weight, **configuration):
        from .state_space import upload_causal_conv

        return upload_causal_conv(self, weight, **configuration)

    def upload_gdn_gates(self, a_log, dt_bias, **configuration):
        from .state_space import upload_gdn_gates

        return upload_gdn_gates(self, a_log, dt_bias, **configuration)

    def upload_gated_rmsnorm(self, weight, **configuration):
        from .state_space import upload_gated_rmsnorm

        return upload_gated_rmsnorm(self, weight, **configuration)

    def upload_zero_centered_rmsnorm(self, weight, **configuration):
        from .state_space import upload_zero_centered_rmsnorm

        return upload_zero_centered_rmsnorm(self, weight, **configuration)

    def prepare_gdn(self, **configuration):
        from .gdn import prepare_gdn

        return prepare_gdn(self, **configuration)

    def upload_qwen_qk_norm(self, query_weight, key_weight, **configuration):
        from .qwen_attention import upload_qwen_qk_norm

        return upload_qwen_qk_norm(self, query_weight, key_weight, **configuration)

    def prepare_qwen_output_gate(self, **configuration):
        from .qwen_attention import prepare_qwen_output_gate

        return prepare_qwen_output_gate(self, **configuration)

    def prepare_rope(self, **configuration):
        from .attention import prepare_rope

        return prepare_rope(self, **configuration)

    def prepare_attention(self, **configuration):
        from .attention import prepare_attention

        return prepare_attention(self, **configuration)


class CudaBackend(_SharedPreparedBackend):
    """Explicit architecture-tagged native library; loading never builds a model."""

    name = "venturi-blocks-cuda"
    execution_kind = "native-cuda"

    @property
    def capabilities(self):
        return _block_capabilities(
            reference=False,
            architecture=self.architecture,
            q4=self.q4_backend is not None,
        )

    def __init__(self, library, *, q4_backend=None):
        self.library = Path(library).expanduser().resolve(strict=True)
        self.q4_backend = q4_backend
        self._native = ctypes.CDLL(str(self.library))
        self._native.venturi_blocks_abi.argtypes = []
        self._native.venturi_blocks_abi.restype = ctypes.c_int
        if self._native.venturi_blocks_abi() != 16:
            raise RuntimeError("Venturi blocks ABI mismatch; rebuild explicitly")
        self._native.venturi_blocks_arch.argtypes = []
        self._native.venturi_blocks_arch.restype = ctypes.c_int
        self.architecture = self._native.venturi_blocks_arch()
        if self.architecture not in (70, 89):
            raise RuntimeError("Unsupported Venturi blocks architecture tag")
        signatures = {
            "venturi_blocks_abi": ([], ctypes.c_int),
            "venturi_blocks_error": ([], ctypes.c_char_p),
            "venturi_greedy_run": (
                [ctypes.c_void_p] * 3
                + [ctypes.c_size_t]
                + [ctypes.c_int] * 3
                + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_fp32_linear_run": (
                [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_dense_create": ([ctypes.c_int] * 4, ctypes.c_void_p),
            "venturi_dense_destroy": ([ctypes.c_void_p], None),
            "venturi_dense_run": (
                [ctypes.c_void_p] * 5 + [ctypes.c_size_t, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_rms_run": (
                [ctypes.c_void_p] * 3
                + [ctypes.c_int] * 3
                + [ctypes.c_float, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_swiglu_run": (
                [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_rope_run": (
                [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_qwen_qk_run": (
                [ctypes.c_void_p] * 8
                + [ctypes.c_int] * 6
                + [ctypes.c_float, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_qwen_gate_run": (
                [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_attention_run": (
                [ctypes.c_void_p] * 4
                + [ctypes.c_int] * 6
                + [ctypes.c_float, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_q8_embedding_run": (
                [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_q8_head_run": (
                [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_embedding_run": (
                [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_gdn_run": (
                [ctypes.c_void_p] * 8
                + [ctypes.c_int] * 8
                + [ctypes.c_float, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_conv_run": (
                [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_gates_run": (
                [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_gated_rms_run": (
                [ctypes.c_void_p] * 4
                + [ctypes.c_int] * 5
                + [ctypes.c_float, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "venturi_add_run": (
                [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self._native, name)
            function.argtypes, function.restype = args, result

    def _check(self, result):
        if result < 0:
            raise RuntimeError(
                self._native.venturi_blocks_error().decode("utf-8", "replace")
            )

    def validate_device(self, dtype, device):
        if dtype not in _DTYPES:
            raise ValueError("Native blocks require FP16, BF16 or FP32")
        target = self.resolve_device(device)
        if self.architecture == 70 and dtype == torch.bfloat16:
            raise ValueError("SM70 requires FP16 or FP32, not native BF16")
        return target

    def resolve_device(self, device):
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("CudaBackend requires an explicit CUDA device")
        with torch.cuda.device(target):
            target = torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Preparation must run outside CUDA graph capture")
            expected = (self.architecture // 10, self.architecture % 10)
            if torch.cuda.get_device_capability(target) != expected:
                raise ValueError(
                    f"Blocks library requires SM{self.architecture}; select a matching explicit build"
                )
        return target

    def upload_linear(self, weight, *, dtype, device):
        if not isinstance(weight, (DenseWeight, Q4RowSplitWeight, Q8RowSplitWeight)):
            raise TypeError(
                "Linear requires explicit DenseWeight, Q4RowSplitWeight or Q8RowSplitWeight"
            )
        weight.validate()
        target = self.validate_device(dtype, device)
        if isinstance(weight, Q8RowSplitWeight):
            from .vocabulary import Q8VocabularyWeights

            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Native Q8 supports FP16/BF16 activations")
            resident = Q8VocabularyWeights(
                weight.codes, weight.scales, weight.in_features
            ).upload(self, device=target, dtype=dtype)
            return ResidentLinear(
                self,
                resident,
                resident.storage_tensors,
                dtype,
                target,
                weight.in_features,
                weight.out_features,
            )
        if isinstance(weight, Q4RowSplitWeight):
            if self.q4_backend is None:
                raise ValueError("Q4 requires an explicitly supplied AdaBackend")
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Q4 activations require float16 or bfloat16")
            resident = self.q4_backend.upload_linear(weight, device=target)
            return ResidentLinear(
                self,
                resident,
                resident.storage_tensors,
                dtype,
                target,
                weight.in_features,
                weight.out_features,
            )
        if dtype != weight.data.dtype:
            raise ValueError(
                "Dense weight and activation dtype must match; no implicit cast"
            )
        with torch.cuda.device(target):
            if not bool(torch.isfinite(weight.data).all()):
                raise ValueError("Linear weights must be finite")
            owned = weight.data.detach().to(target, copy=True)
            torch.cuda.current_stream(target).synchronize()
        return ResidentLinear(
            self,
            owned,
            (owned,),
            dtype,
            target,
            weight.in_features,
            weight.out_features,
        )

    def bind_linear(self, resident, *, tokens):
        _shape(tokens, 1)
        if resident._backend is not self:
            raise ValueError("Resident weight belongs to a different backend instance")
        from .vocabulary import ResidentQ8Vocabulary

        if isinstance(resident._payload, ResidentQ8Vocabulary):
            return _Q8Linear(resident._payload.prepare(tokens=tokens))
        if not isinstance(resident._payload, torch.Tensor):
            return _Q4Linear(
                resident._payload.prepare(tokens=tokens, dtype=resident.dtype)
            )
        if resident._payload.dtype != resident.dtype:
            return _FP32WeightLinear(resident, tokens)
        with torch.cuda.device(resident.device):
            handle = self._native.venturi_dense_create(
                tokens,
                resident.out_features,
                resident.in_features,
                _DTYPES[resident.dtype],
            )
            if not handle:
                self._check(-1)
        return _DenseLinear(self, resident._payload, tokens, handle)

    def upload_rmsnorm(self, weight, *, dtype, device, epsilon):
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 1
            or not weight.is_floating_point()
        ):
            raise ValueError("RMSNorm weight must be a floating-point vector")
        _shape(1, weight.numel())
        epsilon = _epsilon(epsilon)
        target = self.validate_device(dtype, device)
        with torch.cuda.device(target):
            owned = (
                weight.detach()
                .to(device=target, dtype=torch.float32, copy=True)
                .contiguous()
            )
            if not bool(torch.isfinite(owned).all()):
                raise ValueError("RMSNorm weight must be finite in FP32")
            torch.cuda.current_stream(target).synchronize()
        return ResidentRMSNorm(self, owned, dtype, epsilon)

    def bind_rmsnorm(self, resident, *, tokens):
        _shape(tokens, 1)
        if resident._backend is not self:
            raise ValueError("Resident weight belongs to a different backend instance")
        return _RMSNorm(
            self, resident._weight, tokens, resident.dtype, resident.epsilon
        )

    def prepare_swiglu(self, *, tokens, features, dtype, device):
        return _Elementwise(
            self,
            _shape(tokens, features),
            dtype,
            self.validate_device(dtype, device),
            True,
        )

    def prepare_add(self, *, tokens, features, dtype, device):
        return _Elementwise(
            self,
            _shape(tokens, features),
            dtype,
            self.validate_device(dtype, device),
            False,
        )


@dataclass(frozen=True)
class _FP32WeightLinear:
    """Small floating projections, FP32 dot/accumulation, one activation cast.

    Native execution has no numerical Torch fallback, workspace or allocation.
    The direct kernel is a fidelity baseline, not an optimized large GEMM route.
    """

    resident: ResidentLinear
    tokens: int
    workspace_bytes = 0

    @property
    def route(self):
        return (
            "torch-fp32-weight-reference"
            if self.resident._backend.execution_kind == "torch"
            else "venturi-fp32-weight-direct"
        )

    def __post_init__(self):
        _shape(self.tokens, self.resident.in_features)
        if self.tokens * self.resident.out_features >= 2**31:
            raise ValueError("FP32 Linear row count must fit int32")

    @property
    def dtype(self):
        return self.resident.dtype

    @property
    def device(self):
        return self.resident.device

    @property
    def in_features(self):
        return self.resident.in_features

    @property
    def out_features(self):
        return self.resident.out_features

    @property
    def storage_tensors(self):
        return self.resident.storage_tensors

    def run(self, x, *, out, workspace=None):
        r = self.resident
        weight = r.storage_tensors[0]
        if workspace is not None:
            raise ValueError("FP32-parameter direct Linear uses no workspace")
        _tensor(
            weight, (r.out_features, r.in_features), torch.float32, r.device, "weight"
        )
        _tensor(x, (self.tokens, r.in_features), r.dtype, r.device, "x")
        _tensor(out, (self.tokens, r.out_features), r.dtype, r.device, "out")
        _no_overlap(out, (x, weight))
        if r._backend.execution_kind == "torch":
            out.copy_((x.float() @ weight.T).to(r.dtype))
        else:
            with torch.cuda.device(r.device):
                r._backend._check(
                    r._backend._native.venturi_fp32_linear_run(
                        x.data_ptr(),
                        weight.data_ptr(),
                        out.data_ptr(),
                        self.tokens,
                        r.out_features,
                        r.in_features,
                        _DTYPES[r.dtype],
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return out


@dataclass(frozen=True)
class _DenseLinear:
    _backend: CudaBackend
    _weight: torch.Tensor
    tokens: int
    _handle: int
    workspace_bytes = 4 * 1024 * 1024
    route = "cublas-gemm"

    def __post_init__(self):
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(
            self,
            "_finalizer",
            weakref.finalize(
                self, self._backend._native.venturi_dense_destroy, self._handle
            ),
        )

    @property
    def dtype(self):
        return self._weight.dtype

    @property
    def device(self):
        return self._weight.device

    @property
    def in_features(self):
        return self._weight.shape[1]

    @property
    def out_features(self):
        return self._weight.shape[0]

    @property
    def storage_tensors(self):
        return (self._weight,)

    def run(self, x, *, out, workspace=None):
        if not self._finalizer.alive:
            raise RuntimeError("Linear plan has been closed")
        _tensor(x, (self.tokens, self.in_features), self.dtype, self.device, "x")
        _tensor(out, (self.tokens, self.out_features), self.dtype, self.device, "out")
        _tensor(
            workspace, (self.workspace_bytes,), torch.uint8, self.device, "workspace"
        )
        if workspace.data_ptr() % 256:
            raise ValueError("cuBLAS workspace requires 256-byte alignment")
        _no_overlap(out, (x, self._weight, workspace))
        _no_overlap(workspace, (x, self._weight))
        # Handle setters are mutable host state; serialize submission, not GPU work.
        with self._lock, torch.cuda.device(self.device):
            if not self._finalizer.alive:
                raise RuntimeError("Linear plan has been closed")
            self._backend._check(
                self._backend._native.venturi_dense_run(
                    self._handle,
                    x.data_ptr(),
                    self._weight.data_ptr(),
                    out.data_ptr(),
                    workspace.data_ptr(),
                    workspace.numel(),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            )
        return out

    def close(self):
        with self._lock:
            self._finalizer()


@dataclass(frozen=True)
class _Q8Linear:
    plan: object
    workspace_bytes = 0
    route = "venturi-q8-direct"

    @property
    def tokens(self):
        return self.plan.tokens

    @property
    def dtype(self):
        return self.plan.resident.dtype

    @property
    def device(self):
        return self.plan.resident.device

    @property
    def in_features(self):
        return self.plan.resident.hidden_size

    @property
    def out_features(self):
        return self.plan.resident.vocab_size

    @property
    def storage_tensors(self):
        return self.plan.resident.storage_tensors

    def run(self, x, *, out, workspace=None):
        if workspace is not None:
            raise ValueError("Direct Q8 Linear uses no workspace")
        return self.plan.run(x, out=out)


@dataclass(frozen=True)
class _Q4Linear:
    plan: object
    workspace_bytes = 0

    def __getattr__(self, name):
        return getattr(self.plan, name)

    @property
    def storage_tensors(self):
        return (self.plan._weight.codes, self.plan._weight.scales)

    def run(self, x, *, out, workspace=None):
        if workspace is not None:
            raise ValueError("Q4 Linear uses no workspace")
        return self.plan.run(x, out=out)


@dataclass(frozen=True)
class _RMSNorm:
    _backend: CudaBackend
    _weight: torch.Tensor
    tokens: int
    dtype: torch.dtype
    epsilon: float
    workspace_bytes = 0

    @property
    def storage_tensors(self):
        return (self._weight,)

    @property
    def device(self):
        return self._weight.device

    @property
    def features(self):
        return self._weight.numel()

    def run(self, x, *, out):
        shape = (self.tokens, self.features)
        _tensor(x, shape, self.dtype, self.device, "x")
        _tensor(out, shape, self.dtype, self.device, "out")
        _no_overlap(out, (x, self._weight))
        with torch.cuda.device(self.device):
            self._backend._check(
                self._backend._native.venturi_rms_run(
                    x.data_ptr(),
                    self._weight.data_ptr(),
                    out.data_ptr(),
                    self.tokens,
                    self.features,
                    _DTYPES[self.dtype],
                    self.epsilon,
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            )
        return out


@dataclass(frozen=True)
class _Elementwise:
    _backend: CudaBackend
    shape: tuple
    dtype: torch.dtype
    device: torch.device
    swiglu: bool
    workspace_bytes = 0

    def run(self, a, b, *, out):
        for name, tensor in (("a", a), ("b", b), ("out", out)):
            _tensor(tensor, self.shape, self.dtype, self.device, name)
        _no_overlap(out, (a, b), allow_inplace=True)
        with torch.cuda.device(self.device):
            function = (
                self._backend._native.venturi_swiglu_run
                if self.swiglu
                else self._backend._native.venturi_add_run
            )
            self._backend._check(
                function(
                    a.data_ptr(),
                    b.data_ptr(),
                    out.data_ptr(),
                    a.numel(),
                    _DTYPES[self.dtype],
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            )
        return out


class TorchBackend(_SharedPreparedBackend):
    """Explicit CPU/CUDA reference. Uses PyTorch and allocates intermediates."""

    name = "torch-reference"
    execution_kind = "torch"
    capabilities = _block_capabilities(reference=True)

    def validate_device(self, dtype, device):
        if dtype not in (*_DTYPES, torch.float64):
            raise ValueError("Reference operators require floating-point activations")
        return self.resolve_device(device)

    def resolve_device(self, device):
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Reference operators require CPU/CUDA")
        if target.type == "cpu":
            # PyTorch CPU tensors do not retain a parsed cpu:N index.
            return torch.device("cpu")
        if target.type == "cuda":
            with torch.cuda.device(target):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Preparation must run outside CUDA graph capture"
                    )
                target = torch.device("cuda", torch.cuda.current_device())
        return target

    def upload_linear(self, weight, *, dtype, device):
        if not isinstance(weight, (DenseWeight, Q4RowSplitWeight, Q8RowSplitWeight)):
            raise TypeError(
                "Linear requires explicit DenseWeight, Q4RowSplitWeight or Q8RowSplitWeight"
            )
        weight.validate()
        device = self.validate_device(dtype, device)
        if isinstance(weight, DenseWeight):
            if dtype != weight.data.dtype:
                raise ValueError(
                    "Dense weight and activation dtype must match; no implicit cast"
                )
            data = weight.data.detach().to(device, copy=True)
        else:
            # Explicit diagnostic materialization, never a native execution fallback.
            if weight.codes.device.type != "cpu":
                raise ValueError(
                    "Quantized reference preparation requires explicit CPU planes"
                )
            work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            data = weight.dequantize_reference().to(device=device, dtype=work_dtype)
        if not bool(torch.isfinite(data).all()):
            raise ValueError("Linear weights must be finite in the execution dtype")
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        return ResidentLinear(
            self,
            (
                data,
                weight.format
                if isinstance(weight, (Q4RowSplitWeight, Q8RowSplitWeight))
                else False,
            ),
            (data,),
            dtype,
            device,
            weight.in_features,
            weight.out_features,
        )

    def bind_linear(self, resident, *, tokens):
        _shape(tokens, 1)
        if resident._backend is not self:
            raise ValueError("Resident weight belongs to a different backend instance")
        data, quantized = resident._payload
        if not quantized and data.dtype != resident.dtype:
            return _FP32WeightLinear(resident, tokens)
        return (
            _TorchQuantizedLinear(data, tokens, resident.dtype, quantized)
            if quantized
            else _TorchLinear(data, tokens)
        )

    def upload_rmsnorm(self, weight, *, dtype, device, epsilon):
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 1
            or not weight.is_floating_point()
        ):
            raise ValueError("RMSNorm weight must be a floating-point vector")
        _shape(1, weight.numel())
        device = self.validate_device(dtype, device)
        # Match native norm storage precision; FP64 is an explicit reference mode.
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        owned = (
            weight.detach().to(device=device, dtype=work_dtype, copy=True).contiguous()
        )
        if not bool(torch.isfinite(owned).all()):
            raise ValueError("RMSNorm weight must be finite")
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        return ResidentRMSNorm(self, owned, dtype, _epsilon(epsilon))

    def bind_rmsnorm(self, resident, *, tokens):
        _shape(tokens, 1)
        if resident._backend is not self:
            raise ValueError("Resident weight belongs to a different backend instance")
        return _TorchRMS(resident._weight, tokens, resident.dtype, resident.epsilon)

    def prepare_swiglu(self, *, tokens, features, dtype, device):
        return _TorchElementwise(
            _shape(tokens, features), dtype, self.validate_device(dtype, device), True
        )

    def prepare_add(self, *, tokens, features, dtype, device):
        return _TorchElementwise(
            _shape(tokens, features), dtype, self.validate_device(dtype, device), False
        )


@dataclass(frozen=True)
class _TorchLinear:
    _weight: torch.Tensor
    tokens: int
    workspace_bytes = 0
    route = "torch-reference-mm"

    @property
    def dtype(self):
        return self._weight.dtype

    @property
    def storage_tensors(self):
        return (self._weight,)

    @property
    def device(self):
        return self._weight.device

    @property
    def in_features(self):
        return self._weight.shape[1]

    @property
    def out_features(self):
        return self._weight.shape[0]

    def run(self, x, *, out, workspace=None):
        if workspace is not None:
            raise ValueError("Torch reference uses no explicit workspace")
        _tensor(x, (self.tokens, self.in_features), self.dtype, self.device, "x")
        _tensor(out, (self.tokens, self.out_features), self.dtype, self.device, "out")
        _no_overlap(out, (x, self._weight))
        torch.mm(x, self._weight.T, out=out)
        return out


@dataclass(frozen=True)
class _TorchQuantizedLinear:
    """Exact stored Q4/Q8 values materialized in FP32/64 for reference only.

    Rounding decoded weights to BF16 would change the represented weights,
    especially near cancellation. Only each Linear output is activation-cast.
    """

    _weight: torch.Tensor
    tokens: int
    dtype: torch.dtype
    workspace_bytes = 0
    format: str

    @property
    def route(self):
        precision = "fp64" if self._weight.dtype == torch.float64 else "fp32"
        return f"torch-reference-{self.format.split('_')[0]}-{precision}"

    @property
    def device(self):
        return self._weight.device

    @property
    def storage_tensors(self):
        return (self._weight,)

    @property
    def in_features(self):
        return self._weight.shape[1]

    @property
    def out_features(self):
        return self._weight.shape[0]

    def run(self, x, *, out, workspace=None):
        if workspace is not None:
            raise ValueError("Torch quantized reference uses no explicit workspace")
        _tensor(x, (self.tokens, self.in_features), self.dtype, self.device, "x")
        _tensor(out, (self.tokens, self.out_features), self.dtype, self.device, "out")
        _no_overlap(out, (x, self._weight))
        result = x.to(self._weight.dtype) @ self._weight.T
        out.copy_(result.to(self.dtype))
        return out


@dataclass(frozen=True)
class _TorchRMS:
    _weight: torch.Tensor
    tokens: int
    dtype: torch.dtype
    epsilon: float

    @property
    def storage_tensors(self):
        return (self._weight,)

    @property
    def device(self):
        return self._weight.device

    @property
    def features(self):
        return self._weight.numel()

    def run(self, x, *, out):
        for name, tensor in (("x", x), ("out", out)):
            _tensor(tensor, (self.tokens, self.features), self.dtype, self.device, name)
        _no_overlap(out, (x, self._weight))
        stats = x.double() if self.dtype == torch.float64 else x.float()
        scaled = stats * torch.rsqrt(
            stats.square().mean(-1, keepdim=True) + self.epsilon
        )
        out.copy_((scaled * self._weight.to(stats.dtype)).to(self.dtype))
        return out


@dataclass(frozen=True)
class _TorchElementwise:
    shape: tuple
    dtype: torch.dtype
    device: torch.device
    swiglu: bool

    def run(self, a, b, *, out):
        for name, tensor in (("a", a), ("b", b), ("out", out)):
            _tensor(tensor, self.shape, self.dtype, self.device, name)
        _no_overlap(out, (a, b), allow_inplace=True)
        if self.swiglu:
            work = torch.float64 if self.dtype == torch.float64 else torch.float32
            out.copy_((F.silu(a.to(work)) * b.to(work)).to(self.dtype))
        else:
            torch.add(a, b, out=out)
        return out
