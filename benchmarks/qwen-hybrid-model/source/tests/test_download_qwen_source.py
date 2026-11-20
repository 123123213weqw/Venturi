"""Source acquisition metadata guards; tests never download model weights."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def tool():
    path = Path(__file__).resolve().parents[1] / "tools/download_qwen_source.py"
    spec = spec_from_file_location("source_acquisition", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return dict(
        model_type="qwen3_5",
        text_config=dict(
            hidden_size=5120,
            intermediate_size=17408,
            num_hidden_layers=64,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            vocab_size=248320,
            dtype="bfloat16",
        ),
    )


def test_source_geometry_and_precision():
    module, raw = tool(), config()
    assert module.validate_config(raw)["hidden_size"] == 5120
    raw["text_config"]["num_hidden_layers"] = 65
    with pytest.raises(ValueError, match="geometry"):
        module.validate_config(raw)
    raw = config()
    raw["quantization_config"] = {"bits": 4}
    with pytest.raises(ValueError, match="original BF16"):
        module.validate_config(raw)


def test_source_allowlist_and_lfs_metadata():
    module = tool()
    files = [
        SimpleNamespace(
            rfilename="model-1.safetensors", size=128, lfs=SimpleNamespace(sha256="abc")
        ),
        SimpleNamespace(rfilename="model.py", size=100, lfs=None),
        SimpleNamespace(rfilename="config.json", size=10, lfs=None),
    ]
    result = module.selected_files(files)
    assert [r["name"] for r in result] == ["model-1.safetensors", "config.json"]
    assert result[0]["lfs_sha256"] == "abc"
    for name in ("../bad.safetensors", "/tmp/bad.safetensors"):
        with pytest.raises(ValueError, match="filename"):
            module.selected_files([SimpleNamespace(rfilename=name, size=100, lfs=None)])
    with pytest.raises(ValueError, match="No original"):
        module.selected_files(files[1:])


def test_source_atomic_manifest_and_hash(tmp_path):
    module = tool()
    path = tmp_path / "source.json"
    module.atomic_json(path, {"status": "downloading", "revision": "abc"})
    assert json.loads(path.read_text())["revision"] == "abc"
    assert not (tmp_path / "source.json.tmp").exists()
    value = module.sha256(path)
    assert len(value) == 64


def test_download_orchestration_pins_endpoint_and_verifies_lfs(tmp_path, monkeypatch):
    import hashlib
    import sys

    module = tool()
    raw = json.dumps(config()).encode()
    payload = b"test fixture, not a model"
    digest = hashlib.sha256(payload).hexdigest()
    files = [
        SimpleNamespace(rfilename="config.json", size=len(raw), lfs=None),
        SimpleNamespace(
            rfilename="model.safetensors",
            size=len(payload),
            lfs=SimpleNamespace(sha256=digest),
        ),
    ]
    calls = []

    class Api:
        def __init__(self, *, endpoint):
            assert endpoint == "https://hf-mirror.com"

        def model_info(self, repo, *, revision, files_metadata, token, timeout):
            assert (
                repo == "Qwen/Qwen3.5-27B"
                and files_metadata
                and token is False
                and timeout == 20
            )
            calls.append(revision)
            return SimpleNamespace(sha="c" * 40, siblings=files)

    def hf_download(repo, filename, *, revision, local_dir, token, endpoint):
        assert (
            revision == "c" * 40
            and token is False
            and endpoint == "https://hf-mirror.com"
        )
        (local_dir / filename).write_bytes(raw)
        return str(local_dir / filename)

    def snapshot(
        repo, *, revision, local_dir, allow_patterns, max_workers, token, endpoint
    ):
        assert (
            revision == "c" * 40
            and token is False
            and max_workers == 2
            and endpoint == "https://hf-mirror.com"
        )
        assert "*.safetensors" in allow_patterns
        (local_dir / "model.safetensors").write_bytes(payload)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=Api, hf_hub_download=hf_download, snapshot_download=snapshot
        ),
    )
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda p: SimpleNamespace(free=2**40)
    )
    args = ["--output", str(tmp_path / "source"), "--endpoint", "https://hf-mirror.com"]
    module.main(args)
    result = json.loads((tmp_path / "source/venturi-source.json").read_text())
    assert result["status"] == "complete" and result["revision"] == "c" * 40
    assert result["files"][1]["actual_sha256"] == digest
    module.main(args)
    assert calls == ["main", "c" * 40]

    with pytest.raises(ValueError, match="different repository"):
        module.main(args + ["--repo", "Other/Model"])
    assert (
        json.loads((tmp_path / "source/venturi-source.json").read_text())["status"]
        == "complete"
    )
