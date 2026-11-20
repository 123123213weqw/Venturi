"""Explicit research plans; no automatic model or ordinary-operator dispatch."""

import ctypes as C
from dataclasses import dataclass
import torch
from venturi.ops import prepare_q8_dp4a
from venturi.ops.q8_shared import PreparedQ8Shared
from venturi.ops.q8_fused import prepare_q8_fused, OutputDescriptor


def plain(r, rows, staging, passes=2):
    p = prepare_q8_dp4a(r, tokens=3, residual_compensation=passes == 2)
    if staging == 0 and p.workspace_bytes > 48 * 1024:
        raise ValueError("Full T3 activation exceeds 48KiB")
    fn = p.resident.backend._native.venturi_q8_t3_run
    fn.argtypes = (
        [C.c_void_p, C.c_uint64] + [C.c_void_p] * 3 + [C.c_int] * 7 + [C.c_void_p]
    )
    fn.restype = C.c_int
    return PreparedQ8Shared(p, rows, fn, staging)


@dataclass
class Fused:
    old: object
    rows: int
    staging: int
    fn: object

    @property
    def workspace_bytes(self):
        return self.old.workspace_bytes

    def run(self, x, *, outputs, workspace):
        desc = self.old._arguments(x, outputs, workspace)
        p, r = self.old.base, self.old.residents[0]
        with torch.cuda.device(r.device):
            stream = torch.cuda.current_stream(r.device).cuda_stream
            r.backend._check(
                p._native.quantize(
                    x.data_ptr(),
                    workspace.data_ptr(),
                    p.workspace_bytes,
                    3,
                    r.hidden_size,
                    p.passes,
                    stream,
                )
            )
            r.backend._check(
                self.fn(
                    workspace.data_ptr(),
                    p.workspace_bytes,
                    desc,
                    len(outputs),
                    3,
                    r.hidden_size,
                    r._codes.shape[1],
                    p.passes,
                    self.rows,
                    self.staging,
                    stream,
                )
            )
        return outputs


def fused(rs, rows, staging, passes=2, strides=None):
    old = prepare_q8_fused(
        rs, tokens=3, residual_compensation=passes == 2, output_strides=strides
    )
    if staging == 0 and old.workspace_bytes > 48 * 1024:
        raise ValueError("Full T3 activation exceeds 48KiB")
    fn = old.residents[0].backend._native.venturi_q8_t3_fused_run
    fn.argtypes = (
        [C.c_void_p, C.c_uint64, C.POINTER(OutputDescriptor)]
        + [C.c_int] * 7
        + [C.c_void_p]
    )
    fn.restype = C.c_int
    return Fused(old, rows, staging, fn)
