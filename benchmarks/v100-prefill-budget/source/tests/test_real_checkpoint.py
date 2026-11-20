"""Opt-in acceptance for any local safetensors checkpoint; never downloads models."""

import os

import pytest
from venturi import load_checkpoint, verify_loaded_checkpoint


@pytest.mark.cuda
def test_full_checkpoint():
    path = os.environ.get("VENTURI_MODEL_PATH")
    if not path or os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip(
            "Set VENTURI_TEST_CUDA=1 and VENTURI_MODEL_PATH for full-checkpoint acceptance"
        )
    model = load_checkpoint(path, device=os.environ.get("VENTURI_DEVICE", "cuda:0"))
    verification = verify_loaded_checkpoint(model)
    assert verification.verified_tensors == len(model.weights)
    assert verification.verified_bytes == model.stats.tensor_bytes
