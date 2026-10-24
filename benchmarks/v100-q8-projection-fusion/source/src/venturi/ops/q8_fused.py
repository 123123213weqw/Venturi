"""Explicit research fusion of complete Q8 row dots sharing one activation."""

import ctypes as C
from dataclasses import dataclass

import torch
from .blocks import ResidentLinear, _tensor, _no_overlap
from .q8_dp4a import prepare_q8_dp4a


class OutputDescriptor(C.Structure):
    _fields_ = [
        ("codes", C.c_void_p),
        ("scales", C.c_void_p),
        ("out", C.c_void_p),
        ("rows", C.c_int),
        ("stride", C.c_int),
    ]


@dataclass(frozen=True)
class PreparedQ8Fused:
    residents: tuple
    base: object
    strides: tuple
    _run: object

    @property
    def workspace_bytes(self):
        return self.base.workspace_bytes

    @property
    def precision_profile(self):
        return self.base.precision_profile

    def _arguments(self, x, outputs, workspace):
        p = self.base
        r = self.residents[0]
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
            self.residents
        ):
            raise ValueError("one output per fused matrix required")
        _tensor(x, (p.tokens, r.hidden_size), torch.float16, r.device, "x")
        _tensor(workspace, (p.workspace_bytes,), torch.uint8, r.device, "workspace")
        if workspace.data_ptr() % 16:
            raise ValueError("DP4A workspace requires 16-byte alignment")
        storage = tuple(
            t for resident in self.residents for t in resident.storage_tensors
        )
        _no_overlap(workspace, (x,) + storage)
        for i, (out, stride) in enumerate(zip(outputs, self.strides)):
            _tensor(out, (p.tokens, stride), torch.float16, r.device, "out")
            _no_overlap(out, (x, workspace) + storage + tuple(outputs[:i]))
        return (OutputDescriptor * len(outputs))(
            *[
                OutputDescriptor(
                    r._codes.data_ptr(),
                    r._scales.data_ptr(),
                    out.data_ptr(),
                    r.vocab_size,
                    stride,
                )
                for r, out, stride in zip(self.residents, outputs, self.strides)
            ]
        )

    def run(self, x, *, outputs, workspace, packed=False):
        """packed=True is explicit caller-owned quantization reuse, not a pointer cache."""
        if type(packed) is not bool:
            raise ValueError("packed must be boolean")
        descriptors = self._arguments(x, outputs, workspace)
        p = self.base
        r = self.residents[0]
        with torch.cuda.device(r.device):
            stream = torch.cuda.current_stream(r.device).cuda_stream
            if not packed:
                r.backend._check(
                    p._native.quantize(
                        x.data_ptr(),
                        workspace.data_ptr(),
                        p.workspace_bytes,
                        p.tokens,
                        r.hidden_size,
                        p.passes,
                        stream,
                    )
                )
            r.backend._check(
                self._run(
                    workspace.data_ptr(),
                    p.workspace_bytes,
                    descriptors,
                    len(self.residents),
                    p.tokens,
                    r.hidden_size,
                    r._codes.shape[1],
                    p.passes,
                    stream,
                )
            )
        return outputs


def prepare_q8_fused(residents, *, tokens, residual_compensation, output_strides=None):
    if not isinstance(residents, (tuple, list)) or len(residents) not in (2, 3):
        raise ValueError("fused projection requires two or three resident matrices")
    values = tuple(
        r._payload if isinstance(r, ResidentLinear) else r for r in residents
    )
    plans = [
        prepare_q8_dp4a(r, tokens=tokens, residual_compensation=residual_compensation)
        for r in values
    ]
    first = values[0]
    if any(
        r.device != first.device
        or r.hidden_size != first.hidden_size
        or r.backend is not first.backend
        for r in values
    ):
        raise ValueError("fused matrices require the same device, backend and K")
    if sum(r.vocab_size for r in values) >= 2**31 - 48:
        raise ValueError("fused row sum exceeds int32")
    strides = (
        tuple(r.vocab_size for r in values)
        if output_strides is None
        else tuple(output_strides)
    )
    if len(strides) != len(values) or any(
        type(s) is not int
        or s < r.vocab_size
        or s >= 2**31
        or (tokens - 1) * s + r.vocab_size >= 2**31
        for r, s in zip(values, strides)
    ):
        raise ValueError("invalid fused output strides")
    try:
        fn = first.backend._native.venturi_q8_fused_run
    except AttributeError as e:
        raise RuntimeError("Fused projection export absent; rebuild explicitly") from e
    fn.argtypes = (
        [C.c_void_p, C.c_uint64, C.POINTER(OutputDescriptor)]
        + [C.c_int] * 5
        + [C.c_void_p]
    )
    fn.restype = C.c_int
    return PreparedQ8Fused(values, plans[0], strides, fn)
