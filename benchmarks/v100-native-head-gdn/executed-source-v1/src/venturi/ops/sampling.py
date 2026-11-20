"""Prepared exact greedy selection, not stochastic sampling or a decode scheduler.

Native execution is asynchronous, allocation-free and graph-capturable. Results
are [token, status] int64 pairs per independent row. Status 0 means valid; any
NaN/infinity yields [-1, 1]. Callers must check status before using a token. No
host scalar read/exception can report a data-dependent error during capture.
Ties (including signed zero) choose the lowest vocabulary index.
"""

from dataclasses import dataclass

import torch

from .blocks import _DTYPES, _no_overlap, _shape, _tensor


def prepare_greedy(
    backend, *, rows=1, vocab_size, dtype=torch.float32, device="cuda:0"
):
    _shape(rows, vocab_size)
    if rows > 65535:
        raise ValueError("Greedy selection supports at most 65535 independent rows")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Greedy logits require explicit FP16/BF16/FP32")
    target = backend._device(dtype, device)
    return PreparedGreedy(backend, rows, vocab_size, dtype, target)


@dataclass(frozen=True)
class PreparedGreedy:
    """Stateless plan; each concurrent call needs independent output/workspace.

    Caller retains operands/plan/workspace through queued work and graph replay.
    run() never copies logits to CPU, synchronizes, or performs host argmax.
    TorchBackend is an explicitly allocating reference, not a native fallback.
    """

    backend: object
    rows: int
    vocab_size: int
    dtype: torch.dtype
    device: torch.device

    @property
    def workspace_bytes(self):
        return self.rows * min((self.vocab_size + 4095) // 4096, 128) * 16

    def run(self, logits, *, out, workspace):
        _tensor(logits, (self.rows, self.vocab_size), self.dtype, self.device, "logits")
        _tensor(out, (self.rows, 2), torch.int64, self.device, "out")
        _tensor(
            workspace, (self.workspace_bytes,), torch.uint8, self.device, "workspace"
        )
        if workspace.data_ptr() % 16:
            raise ValueError("Greedy workspace must have 16-byte alignment")
        _no_overlap(out, (logits, workspace))
        _no_overlap(workspace, (logits,))
        if self.backend.name == "torch-reference":
            valid = logits.isfinite().all(dim=1)
            out[:, 0].copy_(torch.where(valid, logits.argmax(dim=1), -1))
            out[:, 1].copy_(~valid)
        else:
            with torch.cuda.device(self.device):
                self.backend._check(
                    self.backend._native.venturi_greedy_run(
                        logits.data_ptr(),
                        out.data_ptr(),
                        workspace.data_ptr(),
                        self.workspace_bytes,
                        self.rows,
                        self.vocab_size,
                        _DTYPES[self.dtype],
                        torch.cuda.current_stream(self.device).cuda_stream,
                    )
                )
        return out
