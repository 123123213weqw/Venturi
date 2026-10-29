"""Explicit prebuilt SM70 W4A16 operators. No Torch JIT or automatic build."""
import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch

from .weights import Q4RowSplitWeight


@dataclass(frozen=True)
class PreparedQ4:
    weight: Q4RowSplitWeight
    ready: object


class NativeQ4:
    def __init__(self, library, *, max_output_bytes=64*2**20):
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError('Positive explicit output budget required')
        self.max_output_bytes = max_output_bytes
        self.library = Path(library).resolve(strict=True)
        self.lib = ctypes.CDLL(str(self.library))
        self.lib.venturi_q4_abi.restype = ctypes.c_int
        if self.lib.venturi_q4_abi() != 1:
            raise ValueError('Unsupported Q4 operator ABI')
        self._linear, self._decode = self.lib.venturi_q4_linear, self.lib.venturi_q4_decode
        self._linear.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_void_p]
        self._decode.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
        self._linear.restype = self._decode.restype = ctypes.c_int

    def prepare(self, weight, *, device):
        if not isinstance(weight,Q4RowSplitWeight) or weight.codes.device.type != 'cpu':
            raise ValueError('Explicit CPU Q4 descriptor required for preparation')
        weight.validate()
        if not torch.isfinite(weight.scales).all() or (weight.scales < 0).any() or (
                weight.out_features*weight.in_features*2 > self.max_output_bytes):
            raise ValueError('Finite nonnegative Q4 scales and bounded matrix required')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Explicit CUDA device required')
        with torch.cuda.device(device):
            if torch.cuda.get_device_capability(device) != (7,0):
                raise ValueError('This operator is qualified only for SM70')
            result = Q4RowSplitWeight(weight.codes.to(device),weight.scales.to(device),weight.in_features)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
        return PreparedQ4(result,ready)

    def _lease(self, prepared):
        if not isinstance(prepared,PreparedQ4) or prepared.weight.codes.device.type != 'cuda':
            raise ValueError('Explicit prepared CUDA Q4 weight required')
        w = prepared.weight
        w.validate()
        stream = torch.cuda.current_stream(w.codes.device)
        stream.wait_event(prepared.ready)
        w.codes.record_stream(stream)
        w.scales.record_stream(stream)
        return w,stream

    def decode(self, prepared):
        with torch.cuda.device(prepared.weight.codes.device):
            w,stream = self._lease(prepared)
            if w.out_features*w.in_features*2 > self.max_output_bytes:
                raise ValueError('Q4 decode exceeds output budget')
            out = torch.empty((w.out_features,w.in_features),device=w.codes.device,dtype=torch.float16)
            status = self._decode(w.codes.data_ptr(),w.scales.data_ptr(),out.data_ptr(),w.out_features,
                                  w.in_features,w.padded_k,stream.cuda_stream)
            if status:
                raise RuntimeError(f'Q4 decode CUDA launch failed: {status}')
            return out

    def linear(self, x, prepared):
        if not isinstance(prepared,PreparedQ4):
            raise ValueError('Explicit prepared CUDA Q4 weight required')
        w = prepared.weight
        if x.ndim != 2 or not 1 <= len(x) <= 64 or x.shape[1] != w.in_features or (
                x.dtype != torch.float16 or x.device != w.codes.device or not x.is_contiguous()):
            raise ValueError('Bounded contiguous FP16 CUDA input required')
        if len(x)*w.out_features*2 > self.max_output_bytes:
            raise ValueError('Q4 linear exceeds output budget')
        with torch.cuda.device(x.device):
            w,stream = self._lease(prepared)
            out = torch.empty((len(x),w.out_features),device=x.device,dtype=x.dtype)
            status = self._linear(x.data_ptr(),w.codes.data_ptr(),w.scales.data_ptr(),out.data_ptr(),
                                  len(x),w.out_features,w.in_features,w.padded_k,stream.cuda_stream)
            if status:
                raise RuntimeError(f'Q4 GEMV CUDA launch failed: {status}')
            return out
