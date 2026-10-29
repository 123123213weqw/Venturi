"""CPU-only admission tests; never compile or load CUDA automatically."""

import pytest
import torch

from venturi.ops.flash_next_fp8 import NativeBlockFP8Decoder
from venturi.models.flash_next_cuda_fp8 import CUDABlockFP8ExpertStore


def test_explicit_library_and_output_budget_required(tmp_path):
    for budget in (True, 0, -1):
        with pytest.raises(ValueError):
            NativeBlockFP8Decoder(tmp_path/'not-a-library.so', max_output_bytes=budget)
    with pytest.raises(FileNotFoundError):
        NativeBlockFP8Decoder(tmp_path/'not-a-library.so')
    with pytest.raises(ValueError, match='prebuilt'):
        CUDABlockFP8ExpertStore(None, None, 'experts', decoder=None)


def test_decode_admission_never_calls_native_for_cpu_or_invalid_dtype():
    decoder = NativeBlockFP8Decoder.__new__(NativeBlockFP8Decoder)
    for pairs, dtype in [((), torch.float32), (((torch.ones(1, 1, dtype=torch.uint8),
        torch.ones(1, 1, dtype=torch.int16)),), torch.float32), (((torch.ones(1), torch.ones(1)),), torch.int8)]:
        with pytest.raises(ValueError):
            decoder.decode_many(pairs, dtype=dtype)
