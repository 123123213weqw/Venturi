"""Standalone SM70 research pilot, never selected by production model dispatch."""

import ctypes as C
from dataclasses import dataclass

import torch

from .blocks import ResidentLinear, _no_overlap, _tensor
from .q8_dp4a import prepare_q8_dp4a
from .q8_fused import OutputDescriptor


def workspace_bytes(tokens, k):
    if (
        type(tokens) is not int
        or not 0 < tokens <= 262140
        or type(k) is not int
        or not 0 < k <= 2**31 - 1 - 1024
    ):
        raise ValueError("Invalid DP2A geometry")
    count = tokens * ((k + 31) // 32)
    if count > (2**31 - 1) // 64:
        raise ValueError("DP2A workspace exceeds bounded indexing")
    return count * 68


@dataclass(frozen=True)
class PreparedDP2A:
    pilot: object
    residents: tuple
    tokens: int
    strides: tuple
    dtype: object

    @property
    def workspace_bytes(self):
        return workspace_bytes(self.tokens, self.residents[0].hidden_size)

    def run(self, x, *, outputs, workspace, packed=False):
        """packed=True explicitly reuses caller-owned codes, never a pointer cache."""
        if type(packed) is not bool:
            raise ValueError("packed must be boolean")
        r = self.residents[0]
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(
            self.residents
        ):
            raise ValueError("One output per DP2A matrix required")
        _tensor(x, (self.tokens, r.hidden_size), torch.float16, r.device, "x")
        _tensor(workspace, (self.workspace_bytes,), torch.uint8, r.device, "workspace")
        if workspace.data_ptr() % 16:
            raise ValueError("DP2A workspace needs 16-byte alignment")
        storage = tuple(
            v for resident in self.residents for v in resident.storage_tensors
        )
        _no_overlap(workspace, (x,) + storage)
        for i, (out, stride) in enumerate(zip(outputs, self.strides)):
            _tensor(out, (self.tokens, stride), self.dtype, r.device, "out")
            _no_overlap(out, (x, workspace) + storage + tuple(outputs[:i]))
        descriptors = (OutputDescriptor * len(outputs))(
            *[
                OutputDescriptor(
                    v._codes.data_ptr(),
                    v._scales.data_ptr(),
                    out.data_ptr(),
                    v.vocab_size,
                    stride,
                )
                for v, out, stride in zip(self.residents, outputs, self.strides)
            ]
        )
        with torch.cuda.device(r.device):
            stream = torch.cuda.current_stream(r.device).cuda_stream
            if not packed:
                self.pilot.check(
                    self.pilot.quantize(
                        x.data_ptr(),
                        workspace.data_ptr(),
                        self.workspace_bytes,
                        self.tokens,
                        r.hidden_size,
                        stream,
                    )
                )
            self.pilot.check(
                self.pilot.project(
                    workspace.data_ptr(),
                    self.workspace_bytes,
                    descriptors,
                    len(outputs),
                    self.tokens,
                    r.hidden_size,
                    r._codes.shape[1],
                    int(self.dtype == torch.float32),
                    stream,
                )
            )
        return outputs


class DP2APilot:
    def __init__(self, library):
        self.native = C.CDLL(str(library))
        try:
            self.quantize = self.native.venturi_dp2a_quantize
            self.project = self.native.venturi_dp2a_run
            self.error = self.native.venturi_dp2a_error
            self.integer = self.native.venturi_dp2a_integer_dot
            self.bytes = self.native.venturi_dp2a_bytes
        except AttributeError as e:
            raise RuntimeError(
                "Standalone DP2A exports absent; compile remotely explicitly"
            ) from e
        self.quantize.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_uint64,
            C.c_int,
            C.c_int,
            C.c_void_p,
        ]
        self.quantize.restype = C.c_int
        self.project.argtypes = (
            [C.c_void_p, C.c_uint64, C.POINTER(OutputDescriptor)]
            + [C.c_int] * 5
            + [C.c_void_p]
        )
        self.project.restype = C.c_int
        self.error.argtypes, self.error.restype = [], C.c_char_p
        self.integer.argtypes = [C.c_void_p] * 3 + [C.c_int, C.c_void_p]
        self.integer.restype = C.c_int
        self.bytes.argtypes, self.bytes.restype = [C.c_int, C.c_int], C.c_uint64

    def check(self, status):
        if status:
            raise RuntimeError(self.error().decode())

    def prepare(self, residents, *, tokens, dtype=torch.float16, output_strides=None):
        if not isinstance(residents, (tuple, list)) or not 1 <= len(residents) <= 3:
            raise ValueError("DP2A needs one to three resident Q8 matrices")
        if dtype not in (torch.float16, torch.float32):
            raise ValueError("DP2A output must be FP16 or FP32")
        values = tuple(
            v._payload if isinstance(v, ResidentLinear) else v for v in residents
        )
        for v in values:
            prepare_q8_dp4a(v, tokens=tokens, residual_compensation=True)
        first = values[0]
        needed = workspace_bytes(tokens, first.hidden_size)
        if any(
            v.device != first.device
            or v.hidden_size != first.hidden_size
            or v.backend is not first.backend
            for v in values
        ):
            raise ValueError("DP2A matrices require the same device, backend and K")
        if torch.cuda.get_device_capability(first.device) != (7, 0):
            raise ValueError("DP2A pilot requires explicit SM70")
        strides = (
            tuple(v.vocab_size for v in values)
            if output_strides is None
            else tuple(output_strides)
        )
        if len(strides) != len(values) or any(
            type(s) is not int
            or s < v.vocab_size
            or s >= 2**31
            or (tokens - 1) * s + v.vocab_size >= 2**31
            for s, v in zip(strides, values)
        ):
            raise ValueError("Invalid DP2A strides")
        if sum(v.vocab_size for v in values) >= 2**31 - 48:
            raise ValueError("DP2A row sum exceeds int32")
        if self.bytes(tokens, first.hidden_size) != needed:
            raise RuntimeError("DP2A native workspace contract differs")
        return PreparedDP2A(self, values, tokens, strides, dtype)
