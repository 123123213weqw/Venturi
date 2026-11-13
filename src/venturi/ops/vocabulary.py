"""Independent row-split Q8 vocabulary bindings, lookup and output projection.

The Qwen artifact's embedding/head are Q8, not the Q4 projection format. This
module preserves signed codes/group-32 FP16 scales; it never expands an entire
vocabulary to dense weights in native execution. The simple head kernel is a
correctness baseline, not an optimized prefill GEMM.
"""

from dataclasses import dataclass

import torch

from .blocks import _DTYPES, _no_overlap, _shape, _tensor


def _ids(ids, vocab):
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("Vocabulary IDs must be a nonempty list/tuple")
    if len(ids) >= 2**31 or any(type(i) is not int or not 0 <= i < vocab for i in ids):
        raise ValueError("Vocabulary ID/count is out of range")


@dataclass(frozen=True)
class Q8VocabularyWeights:
    codes: torch.Tensor
    scales: torch.Tensor
    hidden_size: int

    def validate(self):
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("Q8 hidden size must be positive")
        if not isinstance(self.codes, torch.Tensor) or self.codes.ndim != 2:
            raise ValueError("Q8 codes require [vocab,padded_hidden]")
        n, padded = self.codes.shape
        _shape(n, self.hidden_size)
        if padded != (self.hidden_size + 127) // 128 * 128:
            raise ValueError("Q8 hidden padding must be a multiple of 128")
        _tensor(self.codes, (n, padded), torch.int8, torch.device("cpu"), "codes")
        _tensor(
            self.scales, (n, padded // 32), torch.float16, torch.device("cpu"), "scales"
        )
        if bool((self.codes == -128).any()):
            raise ValueError("Q8 code -128 is outside the artifact contract")
        if not bool(torch.isfinite(self.scales).all()) or bool((self.scales < 0).any()):
            raise ValueError("Q8 scales must be finite and nonnegative")

    def upload(self, backend, *, device="cuda:0", dtype=torch.bfloat16):
        self.validate()
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Q8 vocabulary supports FP16/BF16 activations")
        target = backend.validate_device(dtype, device)
        codes, scales = (
            x.detach().to(target, copy=True) for x in (self.codes, self.scales)
        )
        if target.type == "cuda":
            torch.cuda.current_stream(target).synchronize()
        return ResidentQ8Vocabulary(backend, codes, scales, self.hidden_size, dtype)


@dataclass(frozen=True)
class ResidentQ8Vocabulary:
    backend: object
    _codes: torch.Tensor
    _scales: torch.Tensor
    hidden_size: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def device(self):
        return self._codes.device

    @property
    def vocab_size(self):
        return self._codes.shape[0]

    @property
    def storage_tensors(self):
        return (self._codes, self._scales)

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)

    def prepare_embedding(self, ids):
        _ids(ids, self.vocab_size)
        self.backend.validate_device(self.dtype, self.device)
        indices = torch.tensor(ids, dtype=torch.int64, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return Q8Embedding(self, indices)

    def prepare(self, *, tokens):
        _shape(tokens, self.vocab_size)
        if tokens * self.vocab_size >= 2**31:
            raise ValueError("Q8 head row count must fit int32")
        self.backend.validate_device(self.dtype, self.device)
        return Q8OutputHead(self, tokens)


@dataclass(frozen=True)
class Q8Embedding:
    resident: ResidentQ8Vocabulary
    _indices: torch.Tensor

    def run(self, *, out):
        r = self.resident
        _tensor(out, (self._indices.numel(), r.hidden_size), r.dtype, r.device, "out")
        _no_overlap(out, r.storage_tensors + (self._indices,))
        if r.backend.execution_kind == "torch":
            codes, scales = r._codes[self._indices], r._scales[self._indices]
            out.copy_(
                (codes.float() * scales.float().repeat_interleave(32, -1))[
                    :, : r.hidden_size
                ].to(r.dtype)
            )
        else:
            with torch.cuda.device(r.device):
                r.backend._check(
                    r.backend._native.venturi_q8_embedding_run(
                        r._codes.data_ptr(),
                        r._scales.data_ptr(),
                        self._indices.data_ptr(),
                        out.data_ptr(),
                        self._indices.numel(),
                        r.vocab_size,
                        r.hidden_size,
                        r._codes.shape[1],
                        _DTYPES[r.dtype],
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return out


@dataclass(frozen=True)
class Q8OutputHead:
    resident: ResidentQ8Vocabulary
    tokens: int
    workspace_bytes = 0

    def run(self, x, *, out):
        r = self.resident
        _tensor(x, (self.tokens, r.hidden_size), r.dtype, r.device, "x")
        _tensor(out, (self.tokens, r.vocab_size), r.dtype, r.device, "out")
        _no_overlap(out, (x,) + r.storage_tensors)
        if r.backend.execution_kind == "torch":
            # Explicit allocating, tiled FP32 reference. No full-vocabulary dense
            # reconstruction; the block bound keeps peak scratch under control.
            for start in range(0, r.vocab_size, 512):
                end = min(start + 512, r.vocab_size)
                weight = r._codes[start:end].float() * r._scales[
                    start:end
                ].float().repeat_interleave(32, -1)
                out[:, start:end].copy_(
                    (x.float() @ weight[:, : r.hidden_size].T).to(r.dtype)
                )
        else:
            with torch.cuda.device(r.device):
                r.backend._check(
                    r.backend._native.venturi_q8_head_run(
                        x.data_ptr(),
                        r._codes.data_ptr(),
                        r._scales.data_ptr(),
                        out.data_ptr(),
                        self.tokens,
                        r.vocab_size,
                        r.hidden_size,
                        r._codes.shape[1],
                        _DTYPES[r.dtype],
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return out
