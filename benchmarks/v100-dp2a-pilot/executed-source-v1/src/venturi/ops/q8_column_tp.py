"""Explicit input-column TP primitives, not a native model-parallel backend.

Stored weight codes/scales and residual-A8 quantization are retained. Partial
outputs/transport must remain FP32; summation changes FP32 association relative
to a full-K projection. Caller owns all execution scratch; preparation may copy.
"""

import ctypes as C
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .blocks import ResidentLinear, _no_overlap, _shape, _tensor
from .q8_dp4a import _bind as _bind_quantizer
from .vocabulary import ResidentQ8Vocabulary
from .weights import Q8RowSplitWeight


def slice_q8_columns(weight, start, stop):
    """Copy encoded CPU scale groups without dequantization/requantization."""
    if not isinstance(weight, Q8RowSplitWeight):
        raise ValueError("Column slicing requires a Q8RowSplitWeight")
    weight.validate()
    if weight.codes.device.type != "cpu":
        raise ValueError("Column slicing is explicit CPU preparation")
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= weight.in_features
        or start % 32
        or (stop % 32 and stop != weight.in_features)
    ):
        raise ValueError("Column slices must follow stored scale-group boundaries")
    k = stop - start
    padded = (k + 127) // 128 * 128
    codes = torch.zeros((weight.out_features, padded), dtype=torch.int8)
    scales = torch.zeros((weight.out_features, padded // 32), dtype=torch.float16)
    codes[:, :k].copy_(weight.codes[:, start:stop])
    scales[:, : (k + 31) // 32].copy_(weight.scales[:, start // 32 : (stop + 31) // 32])
    return Q8RowSplitWeight(codes, scales, k)


def _bind(backend):
    try:
        partial = backend._native.venturi_q8_column_partial_run
        partial.argtypes = (
            [C.c_void_p, C.c_uint64] + [C.c_void_p] * 3 + [C.c_int] * 4 + [C.c_void_p]
        )
        partial.restype = C.c_int
        reduce = backend._native.venturi_q8_column_sum
        reduce.argtypes = [C.c_void_p] * 3 + [C.c_int, C.c_void_p]
        reduce.restype = C.c_int
    except AttributeError as e:
        raise RuntimeError(
            "Explicit FP32 column-TP exports absent; use a research build"
        ) from e
    return SimpleNamespace(partial=partial, reduce=reduce)


@dataclass(frozen=True)
class PreparedQ8ColumnPartial:
    resident: ResidentQ8Vocabulary
    tokens: int
    workspace_bytes: int
    _native: object
    _quantizer: object

    precision_profile = "w8-a8-g32-f32scale-residual-a8-f32partial-two-rank-sum"

    def run(self, x, *, out, workspace):
        """Compute one FP32 partial, without allocation, transport or a final cast."""
        r = self.resident
        _tensor(x, (self.tokens, r.hidden_size), torch.float16, r.device, "x")
        _tensor(out, (self.tokens, r.vocab_size), torch.float32, r.device, "out")
        _tensor(workspace, (self.workspace_bytes,), torch.uint8, r.device, "workspace")
        _no_overlap(out, (x, workspace) + r.storage_tensors)
        _no_overlap(workspace, (x,) + r.storage_tensors)
        if workspace.data_ptr() % 16:
            raise ValueError("Column-partial workspace requires 16-byte alignment")
        with torch.cuda.device(r.device):
            stream = torch.cuda.current_stream(r.device).cuda_stream
            r.backend._check(
                self._quantizer.quantize(
                    x.data_ptr(),
                    workspace.data_ptr(),
                    self.workspace_bytes,
                    self.tokens,
                    r.hidden_size,
                    2,
                    stream,
                )
            )
            r.backend._check(
                self._native.partial(
                    workspace.data_ptr(),
                    self.workspace_bytes,
                    r._codes.data_ptr(),
                    r._scales.data_ptr(),
                    out.data_ptr(),
                    self.tokens,
                    r.vocab_size,
                    r.hidden_size,
                    r._codes.shape[1],
                    stream,
                )
            )
        return out

    def reduce(self, local, remote, *, out):
        """Sum two same-device FP32 partials; cast once to FP16, no transport."""
        r = self.resident
        shape = (self.tokens, r.vocab_size)
        _tensor(local, shape, torch.float32, r.device, "local")
        _tensor(remote, shape, torch.float32, r.device, "remote")
        _tensor(out, shape, torch.float16, r.device, "out")
        _no_overlap(out, (local, remote) + r.storage_tensors)
        with torch.cuda.device(r.device):
            stream = torch.cuda.current_stream(r.device).cuda_stream
            r.backend._check(
                self._native.reduce(
                    local.data_ptr(),
                    remote.data_ptr(),
                    out.data_ptr(),
                    self.tokens * r.vocab_size,
                    stream,
                )
            )
        return out


def prepare_q8_column_partial(resident, *, tokens):
    """Explicit SM70 research plan; never inferred from an existing model plan."""
    if isinstance(resident, ResidentLinear):
        resident = resident._payload
    if not isinstance(resident, ResidentQ8Vocabulary):
        raise ValueError("Column partial requires resident native Q8 planes")
    _shape(tokens, resident.hidden_size)
    if resident.dtype != torch.float16 or resident.device.type != "cuda":
        raise ValueError("Column partial requires CUDA FP16 activation storage")
    resident.backend._device(resident.dtype, resident.device)
    if torch.cuda.get_device_capability(resident.device) != (7, 0):
        raise ValueError("Column partial is an explicit SM70 research operator")
    if tokens * resident.vocab_size >= 2**31 - 256:
        raise ValueError("Column-partial output geometry exceeds int32")
    native = _bind(resident.backend)
    quantizer = _bind_quantizer(resident.backend)
    size = quantizer.bytes(tokens, resident.hidden_size, 2)
    if not size:
        resident.backend._check(-1)
    return PreparedQ8ColumnPartial(resident, tokens, size, native, quantizer)
