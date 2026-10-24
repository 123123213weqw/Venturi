from dataclasses import replace
import json

import pytest
import torch

from venturi import CheckpointError, load_checkpoint, verify_loaded_checkpoint
from venturi.cli import main


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("chunk_bytes", [1, 7, 16 * 2**20])
def test_exact_verification(make_checkpoint, sharded, chunk_bytes):
    model = load_checkpoint(
        make_checkpoint(sharded=sharded), device="cpu", loader="safetensors"
    )
    result = verify_loaded_checkpoint(model, chunk_bytes=chunk_bytes)
    assert result.verified_tensors == len(model.weights)
    assert result.verified_bytes == model.stats.tensor_bytes
    assert result.verification_seconds >= 0


def test_nan_payload_scalar_and_empty(make_checkpoint):
    tensors = {
        "nan": torch.tensor([0x7FC00001, 0x7FC00002], dtype=torch.int32).view(
            torch.float32
        ),
        "scalar": torch.tensor(-0.0, dtype=torch.float16),
        "empty": torch.empty(0, 3, dtype=torch.float16),
    }
    model = load_checkpoint(
        make_checkpoint(tensors=tensors), device="cpu", loader="safetensors"
    )
    assert verify_loaded_checkpoint(model, chunk_bytes=3).verified_bytes == 10
    model.weights["nan"].view(torch.int32)[0] = 0x7FC00002
    with pytest.raises(CheckpointError, match="nan at storage byte 0"):
        verify_loaded_checkpoint(model)


@pytest.mark.parametrize("mutation", ["bytes", "shape", "dtype", "names", "layout"])
def test_corruption_is_detected(make_checkpoint, mutation):
    model = load_checkpoint(make_checkpoint(), device="cpu", loader="safetensors")
    if mutation == "bytes":
        model.weights["quant.packed_uint4"][2] = 0
    elif mutation == "layout":
        model.weights["quant.packed_int4"].transpose_(0, 1)
    else:
        weights = dict(model.weights)
        if mutation == "shape":
            weights["arbitrary.weight"] = weights["arbitrary.weight"].reshape(-1)
        elif mutation == "dtype":
            weights["arbitrary.weight"] = weights["arbitrary.weight"].float()
        else:
            weights.pop("scalar")
        model = replace(model, weights=weights)
    with pytest.raises(CheckpointError, match="Verification|tensor names"):
        verify_loaded_checkpoint(model, chunk_bytes=2)


@pytest.mark.parametrize("chunk_bytes", [0, -1, True, 1.5])
def test_invalid_chunk_size(make_checkpoint, chunk_bytes):
    model = load_checkpoint(make_checkpoint(), device="cpu", loader="safetensors")
    with pytest.raises(ValueError, match="positive integer"):
        verify_loaded_checkpoint(model, chunk_bytes=chunk_bytes)


@pytest.mark.parametrize("sharded", [False, True])
def test_fp16_contract_preserves_integer_buffers(make_checkpoint, weights, sharded):
    tensors = {
        name: value.half() if value.is_floating_point() else value
        for name, value in weights.items()
    }
    model = load_checkpoint(
        make_checkpoint(tensors=tensors, sharded=sharded),
        device="cpu",
        loader="safetensors",
        require_float_dtype="F16",
    )
    assert model.weights["mask"].dtype == torch.bool
    assert model.weights["positions"].dtype == torch.int64
    assert all(
        t.dtype == torch.float16
        for t in model.weights.values()
        if t.is_floating_point()
    )
    assert verify_loaded_checkpoint(model).verified_tensors == len(tensors)


@pytest.mark.parametrize("required", ["F16", "invalid"])
def test_dtype_contract_fails_before_cuda(make_checkpoint, monkeypatch, required):
    def forbidden():
        pytest.fail("dtype contract must fail before CUDA setup")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(ValueError, match="floating-point storage|require_float_dtype"):
        load_checkpoint(make_checkpoint(), require_float_dtype=required)


def test_verify_cli(make_checkpoint, capsys):
    path = str(make_checkpoint())
    assert (
        main(
            [
                "verify",
                path,
                "--device",
                "cpu",
                "--loader",
                "safetensors",
                "--chunk-bytes",
                "7",
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["verification"]["passed"] is True
    assert output["verification"]["verified_bytes"] == output["tensor_bytes"]
    assert output["verification"]["verified_tensors"] == output["tensor_count"]
    assert "cuda_allocator" not in output
    assert main(["verify", path, "--device", "cpu", "--loader", "safetensors"]) == 0
    assert "Verified 10 tensors" in capsys.readouterr().out
    assert (
        main(
            [
                "verify",
                path,
                "--device",
                "cpu",
                "--loader",
                "safetensors",
                "--require-float-dtype",
                "F16",
            ]
        )
        == 1
    )
    assert "No dtype conversion" in capsys.readouterr().err


def test_acceptance_tool(make_checkpoint, weights, tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    tensors = {
        name: value.half() if value.is_floating_point() else value
        for name, value in weights.items()
    }
    checkpoint = make_checkpoint(tensors=tensors)
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools/run_fp16_acceptance.py"),
            str(checkpoint),
            "--device",
            "cpu",
            "--loaders",
            "safetensors",
            "--runs",
            "1",
            "--output",
            str(output),
        ],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["summary"]["safetensors"]["all_verified"] is True
    assert report["runs"][0]["model_dir"] == "<checkpoint>"
    assert (
        report["runs"][0]["verification"]["verified_bytes"]
        == report["runs"][0]["tensor_bytes"]
    )
