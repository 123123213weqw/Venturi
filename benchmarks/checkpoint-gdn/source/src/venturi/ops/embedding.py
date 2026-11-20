"""Dense embedding lookup with validated host IDs and shared head/lookup storage."""

from dataclasses import dataclass

import torch

from .blocks import _DTYPES, _no_overlap, _tensor


def prepare_embedding(resident, token_ids):
    # Preparation copies only IDs, never the resident vocabulary matrix. IDs are
    # deliberately host-validated before upload, so execution needs no GPU sync.
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        raise ValueError("Embedding token IDs must be a nonempty list/tuple")
    if any(type(i) is not int or not 0 <= i < resident.out_features for i in token_ids):
        raise ValueError("Embedding token ID is outside the vocabulary")
    if len(token_ids) >= 2**31:
        raise ValueError("Embedding token count must fit int32")
    backend = resident._backend
    device = backend._device(resident.dtype, resident.device)
    payload = resident._payload
    if backend.name == "torch-reference":
        weight, quantized = payload
        if quantized:
            raise ValueError("Embedding requires dense storage, not a Q4 Linear")
    else:
        if not isinstance(payload, torch.Tensor):
            raise ValueError("Embedding requires dense storage, not a Q4 Linear")
        weight = payload
    indices = torch.tensor(token_ids, dtype=torch.int64, device=device)
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    return PreparedEmbedding(resident, weight, indices, len(token_ids))


@dataclass(frozen=True)
class PreparedEmbedding:
    """Immutable-by-contract IDs; retain this plan until queued work completes.

    run() only gathers. Do not mutate the private uploaded ID buffer or weights.
    Native execution has no allocation, PyTorch numerical fallback, or ID readback.
    """

    resident: object
    _weight: torch.Tensor
    _indices: torch.Tensor
    tokens: int

    def run(self, *, out):
        r = self.resident
        _tensor(out, (self.tokens, r.in_features), r.dtype, r.device, "out")
        _no_overlap(out, (self._weight, self._indices))
        if r._backend.name == "torch-reference":
            torch.index_select(self._weight, 0, self._indices, out=out)
        else:
            with torch.cuda.device(r.device):
                r._backend._check(
                    r._backend._native.venturi_embedding_run(
                        self._weight.data_ptr(),
                        self._indices.data_ptr(),
                        out.data_ptr(),
                        self.tokens,
                        r.out_features,
                        r.in_features,
                        _DTYPES[r.dtype],
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return out
