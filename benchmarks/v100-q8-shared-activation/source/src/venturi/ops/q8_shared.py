"""Research-only byte-preserving SM70 shared-activation DP4A projection."""

import ctypes as C
from dataclasses import dataclass

import torch
from .blocks import _tensor, _no_overlap
from .q8_dp4a import prepare_q8_dp4a


@dataclass(frozen=True)
class PreparedQ8Shared:
    base: object
    rows: int
    _run: object

    @property
    def workspace_bytes(self):
        return self.base.workspace_bytes

    @property
    def precision_profile(self):
        return self.base.precision_profile

    def run(self, x, *, out, workspace):
        p = self.base
        r = p.resident
        _tensor(x, (p.tokens, r.hidden_size), torch.float16, r.device, "x")
        _tensor(out, (p.tokens, r.vocab_size), torch.float16, r.device, "out")
        _tensor(workspace, (p.workspace_bytes,), torch.uint8, r.device, "workspace")
        _no_overlap(out, (x, workspace) + r.storage_tensors)
        _no_overlap(workspace, (x,) + r.storage_tensors)
        if workspace.data_ptr() % 16:
            raise ValueError("DP4A workspace requires 16-byte alignment")
        with torch.cuda.device(r.device):
            stream = torch.cuda.current_stream(r.device).cuda_stream
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
                    r._codes.data_ptr(),
                    r._scales.data_ptr(),
                    out.data_ptr(),
                    p.tokens,
                    r.vocab_size,
                    r.hidden_size,
                    r._codes.shape[1],
                    p.passes,
                    self.rows,
                    stream,
                )
            )
        return out


def prepare_q8_shared(resident, *, tokens, rows, residual_compensation):
    if (
        type(tokens) is not int
        or tokens not in (1, 2)
        or type(rows) is not int
        or rows not in (4, 8, 16)
    ):
        raise ValueError("Shared DP4A requires T1/T2 and rows 4/8/16")
    p = prepare_q8_dp4a(
        resident, tokens=tokens, residual_compensation=residual_compensation
    )
    if p.workspace_bytes > 48 * 1024:
        raise ValueError("Shared DP4A requires at most 48KiB activation data")
    try:
        fn = p.resident.backend._native.venturi_q8_shared_run
    except AttributeError as e:
        raise RuntimeError(
            "Research shared-activation export absent; rebuild explicitly"
        ) from e
    fn.argtypes = (
        [C.c_void_p, C.c_uint64] + [C.c_void_p] * 3 + [C.c_int] * 6 + [C.c_void_p]
    )
    fn.restype = C.c_int
    return PreparedQ8Shared(p, rows, fn)
