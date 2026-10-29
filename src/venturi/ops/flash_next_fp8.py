"""Explicit prebuilt CUDA byte-decoder library; NEVER auto-build on import.

Works without GPU FP8 or BF16 arithmetic: E4M3FN/BF16 are decoded by integer
bits and FP32 multiplication. No native model/GEMM/activation-quantized backend.
"""

import ctypes
from pathlib import Path

import torch


class NativeBlockFP8Decoder:
    def __init__(self, library, *, max_output_bytes=64*2**20):
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError('Explicit positive output byte budget required')
        self.library = Path(library).resolve(strict=True)
        self.max_output_bytes = max_output_bytes
        self._native = ctypes.CDLL(str(self.library))
        self._native.venturi_fp8_decode_abi.restype = ctypes.c_int
        if self._native.venturi_fp8_decode_abi() != 1:
            raise ValueError('Unsupported explicit FP8 byte-decoder ABI')
        self._decode = self._native.venturi_fp8_decode
        self._decode.argtypes = [ctypes.c_void_p]*4+[ctypes.c_longlong]*2+[ctypes.c_int, ctypes.c_void_p]
        self._decode.restype = ctypes.c_int

    def decode_many(self, pairs, *, dtype):
        pairs = tuple(pairs)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16) or not pairs or len(pairs) > 16:
            raise ValueError('Explicit floating dtype and bounded nonempty matrix list required')
        device, total = None, 0
        for codes, scales in pairs:
            if codes.ndim != 2 or min(codes.shape) < 1 or codes.dtype != torch.uint8 or scales.dtype != torch.int16 or (
                    codes.device.type != 'cuda' or scales.device != codes.device or not codes.is_contiguous() or
                    not scales.is_contiguous() or scales.shape != tuple((n+127)//128 for n in codes.shape)):
                raise ValueError('Contiguous CUDA U8 codes and raw BF16-as-I16 block grid required')
            if device is not None and device != codes.device:
                raise ValueError('All matrices must use one CUDA device')
            device = codes.device
            total += codes.numel()*torch.empty((), dtype=dtype, device='cpu').element_size()
        if total > self.max_output_bytes:
            raise ValueError('Native FP8 decoded outputs exceed explicit byte budget')
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError('Validated decoder synchronizes error flags; graph capture is not qualified')
            errors = torch.zeros((), dtype=torch.int32, device=device)
            outputs = []
            stream = torch.cuda.current_stream(device)
            for codes, scales in pairs:
                output = torch.empty(codes.shape, dtype=dtype, device=device)
                status = self._decode(codes.data_ptr(), scales.data_ptr(), output.data_ptr(), errors.data_ptr(),
                    *codes.shape, (torch.float32, torch.float16, torch.bfloat16).index(dtype), stream.cuda_stream)
                if status:
                    raise RuntimeError(f'FP8 CUDA decoder launch failed: status={status}')
                codes.record_stream(stream)
                scales.record_stream(stream)
                outputs.append(output)
            flags = errors.item()  # one deliberate validation boundary for all matrices
            if flags:
                raise ValueError(f'Invalid FP8 weights: flags={flags} (1=NaN code,2=scale,4=overflow)')
            return tuple(outputs)

    def decode(self, codes, scales, *, dtype):
        return self.decode_many(((codes, scales),), dtype=dtype)[0]
