"""Explicit experimental W8/A8 integer-dot operator, optional residual A8 pass.

Never replaces the default FP16-activation Q8 projection. Weights retain their
stored Q8/FP16 scales; activation group-32 scales are FP32. This is a new compute
profile, not exact-order FP16 arithmetic. Caller owns all device scratch.
"""

import ctypes as C
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .blocks import ResidentLinear, _no_overlap, _shape, _tensor
from .vocabulary import ResidentQ8Vocabulary


def _bind(backend):
    signatures = {
        "bytes": ([C.c_int] * 3, C.c_uint64),
        "quantize": (
            [C.c_void_p] * 2 + [C.c_uint64] + [C.c_int] * 3 + [C.c_void_p],
            C.c_int,
        ),
        "run": (
            [C.c_void_p, C.c_uint64] + [C.c_void_p] * 3 + [C.c_int] * 5 + [C.c_void_p],
            C.c_int,
        ),
    }
    result = {}
    try:
        for name, (args, restype) in signatures.items():
            fn = getattr(backend._native, "venturi_q8_dp4a_" + name)
            fn.argtypes, fn.restype = args, restype
            result[name] = fn
    except AttributeError as e:
        raise RuntimeError(
            "Explicit DP4A exports absent; rebuild the blocks library"
        ) from e
    return SimpleNamespace(**result)


@dataclass(frozen=True)
class PreparedQ8DP4A:
    resident: ResidentQ8Vocabulary
    tokens: int
    passes: int
    workspace_bytes: int
    _native: object

    @property
    def precision_profile(self):
        return "w8-a8-g32-f32scale" + ("-residual-a8" if self.passes == 2 else "")

    def run(self, x, *, out, workspace):
        r = self.resident
        _tensor(x, (self.tokens, r.hidden_size), torch.float16, r.device, "x")
        _tensor(out, (self.tokens, r.vocab_size), torch.float16, r.device, "out")
        _tensor(workspace, (self.workspace_bytes,), torch.uint8, r.device, "workspace")
        _no_overlap(out, (x, workspace) + r.storage_tensors)
        _no_overlap(workspace, (x,) + r.storage_tensors)
        if workspace.data_ptr() % 4:
            raise ValueError("DP4A workspace requires 4-byte alignment")
        with torch.cuda.device(r.device):
            stream = torch.cuda.current_stream(r.device).cuda_stream
            r.backend._check(
                self._native.quantize(
                    x.data_ptr(),
                    workspace.data_ptr(),
                    self.workspace_bytes,
                    self.tokens,
                    r.hidden_size,
                    self.passes,
                    stream,
                )
            )
            r.backend._check(
                self._native.run(
                    workspace.data_ptr(),
                    self.workspace_bytes,
                    r._codes.data_ptr(),
                    r._scales.data_ptr(),
                    out.data_ptr(),
                    self.tokens,
                    r.vocab_size,
                    r.hidden_size,
                    r._codes.shape[1],
                    self.passes,
                    stream,
                )
            )
        return out


def prepare_q8_dp4a(resident, *, tokens, residual_compensation):
    """Opt in explicitly; no default plan dispatch or implicit scratch allocation."""
    if isinstance(resident, ResidentLinear):
        resident = resident._payload
    if not isinstance(resident, ResidentQ8Vocabulary):
        raise ValueError("DP4A requires resident native Q8 planes")
    if type(residual_compensation) is not bool:
        raise ValueError("Explicit boolean residual_compensation required")
    _shape(tokens, resident.hidden_size)
    if resident.dtype != torch.float16 or resident.device.type != "cuda":
        raise ValueError("DP4A experimental profile requires CUDA FP16 input/output")
    resident.backend._device(resident.dtype, resident.device)
    if tokens * resident.vocab_size >= 2**31:
        raise ValueError("DP4A output geometry exceeds int32")
    native = _bind(resident.backend)
    passes = 2 if residual_compensation else 1
    size = native.bytes(tokens, resident.hidden_size, passes)
    if not size:
        resident.backend._check(-1)
    return PreparedQ8DP4A(resident, tokens, passes, size, native)
