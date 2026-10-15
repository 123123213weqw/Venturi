"""Compare every loaded storage byte with its checkpoint, in bounded chunks."""

from dataclasses import dataclass
from time import perf_counter

from safetensors import safe_open
import torch

from .checkpoint import CheckpointError
from .loader import LoadedCheckpoint


@dataclass(frozen=True)
class VerificationStats:
    verified_tensors: int
    verified_bytes: int
    verification_seconds: float
    chunk_bytes: int


def verify_loaded_checkpoint(
    model: LoadedCheckpoint, *, chunk_bytes: int = 16 * 2**20
) -> VerificationStats:
    """Verify exact bytes, including NaN payloads, against the CPU reference.

    At most one chunk is copied back from the GPU at a time. Reference tensors
    are mmap-backed; no second full model is loaded onto the GPU. Verification
    time is separate from loading time. A mismatch raises CheckpointError.
    """
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    started = perf_counter()
    seen: set[str] = set()
    verified_bytes = 0
    if model.weights.keys() != model.checkpoint.tensors.keys():
        raise CheckpointError("Loaded tensor names do not match the checkpoint")
    for path in model.checkpoint.files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in seen or name not in model.weights:
                    raise CheckpointError(
                        f"Reference returned an unexpected or duplicate tensor: {name}"
                    )
                actual = model.weights[name]
                expected = reader.get_tensor(name)
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise CheckpointError(
                        f"Verification shape/dtype mismatch for {name}"
                    )
                actual_bytes = actual.reshape(-1).view(torch.uint8)
                expected_bytes = expected.reshape(-1).view(torch.uint8)
                for offset in range(0, expected_bytes.numel(), chunk_bytes):
                    end = min(offset + chunk_bytes, expected_bytes.numel())
                    chunk = actual_bytes[offset:end].cpu()
                    reference = expected_bytes[offset:end]
                    if not torch.equal(chunk, reference):
                        first = torch.nonzero(chunk != reference)[0].item()
                        raise CheckpointError(
                            f"Verification failed for {name} at storage byte {offset + first}"
                        )
                    verified_bytes += end - offset
                seen.add(name)
    if seen != model.weights.keys():
        raise CheckpointError("Reference did not return every loaded tensor")
    return VerificationStats(
        len(seen), verified_bytes, perf_counter() - started, chunk_bytes
    )
