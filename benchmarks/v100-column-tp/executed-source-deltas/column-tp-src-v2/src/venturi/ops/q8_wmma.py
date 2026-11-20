"""Opt-in SM70 FP16 Tensor Core/group32-scale Q8 projection research.

Weight codes and FP16 input values retain their precision; FP32 WMMA/group/split-K
reduction differs from original arithmetic. Never selected by model generation.
"""

import ctypes as C
from dataclasses import dataclass

import torch

from .blocks import ResidentLinear, _no_overlap, _shape, _tensor
from .vocabulary import ResidentQ8Vocabulary


@dataclass(frozen=True)
class PreparedQ8WMMA:
    resident: ResidentQ8Vocabulary
    tokens: int
    split_k_warps: int
    _run: object
    workspace_bytes: int = 0

    @property
    def precision_profile(self):
        return f"w8-f16tc-g32-f32scale-splitk{self.split_k_warps}"

    def run(self, x, *, out, workspace=None):
        r = self.resident
        if workspace is not None:
            raise ValueError("WMMA group32 uses internal shared memory, no workspace")
        _tensor(x, (self.tokens, r.hidden_size), torch.float16, r.device, "x")
        _tensor(out, (self.tokens, r.vocab_size), torch.float16, r.device, "out")
        _no_overlap(out, (x,) + r.storage_tensors)
        with torch.cuda.device(r.device):
            r.backend._check(
                self._run(
                    x.data_ptr(),
                    r._codes.data_ptr(),
                    r._scales.data_ptr(),
                    out.data_ptr(),
                    self.tokens,
                    r.vocab_size,
                    r.hidden_size,
                    r._codes.shape[1],
                    self.split_k_warps,
                    torch.cuda.current_stream(r.device).cuda_stream,
                )
            )
        return out


def prepare_q8_wmma(resident, *, tokens, split_k_warps=4):
    """Explicit new operator/profile, no model dispatch or allocation fallback."""
    if isinstance(resident, ResidentLinear):
        resident = resident._payload
    if not isinstance(resident, ResidentQ8Vocabulary):
        raise ValueError("WMMA requires resident native Q8 planes")
    if type(split_k_warps) is not int or split_k_warps not in (1, 2, 4, 8):
        raise ValueError("Explicit split_k_warps must be 1, 2, 4 or 8")
    _shape(tokens, resident.hidden_size)
    if (
        resident.dtype != torch.float16
        or resident.device.type != "cuda"
        or resident.backend.architecture != 70
    ):
        raise ValueError("WMMA group32 research requires explicit SM70 FP16")
    if (
        tokens > 262140
        or tokens * max(resident.hidden_size, resident.vocab_size) >= 2**31
        or resident.hidden_size > 2**31 - 128
    ):
        raise ValueError("WMMA launch geometry exceeds supported int32/grid bounds")
    resident.backend._device(resident.dtype, resident.device)
    try:
        fn = resident.backend._native.venturi_q8_wmma_run
    except AttributeError as e:
        raise RuntimeError("Explicit WMMA exports absent; rebuild blocks") from e
    fn.argtypes = [C.c_void_p] * 4 + [C.c_int] * 5 + [C.c_void_p]
    fn.restype = C.c_int
    return PreparedQ8WMMA(resident, tokens, split_k_warps, fn)
