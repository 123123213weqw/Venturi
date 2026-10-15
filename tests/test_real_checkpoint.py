"""Opt-in acceptance for any local safetensors checkpoint; never downloads models."""

import os

import pytest
import torch
from safetensors import safe_open

from venturi import load_checkpoint


@pytest.mark.cuda
def test_full_checkpoint():
    path = os.environ.get("VENTURI_MODEL_PATH")
    if not path or os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip(
            "Set VENTURI_TEST_CUDA=1 and VENTURI_MODEL_PATH for full-checkpoint acceptance"
        )
    model = load_checkpoint(path, device=os.environ.get("VENTURI_DEVICE", "cuda:0"))
    checked = set()
    for shard in model.checkpoint.files:
        with safe_open(str(shard), framework="pt", device="cpu") as reference:
            for name in reference.keys():
                actual = model.weights[name].cpu()
                expected = reference.get_tensor(name)
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
                # Compare storage bytes, including NaN payloads and dtypes for
                # which arithmetic comparison kernels may not exist.
                torch.testing.assert_close(
                    actual.reshape(-1).view(torch.uint8),
                    expected.reshape(-1).view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                checked.add(name)
    assert checked == model.weights.keys()
