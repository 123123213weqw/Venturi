"""Exact bounded range-sample reconstruction/resume without real network I/O."""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("rebuild_range_sample_test", tools / "rebuild_range_sample.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = json.dumps({"a": dict(dtype="BF16", shape=[2], data_offsets=[0, 4]),
                      "b": dict(dtype="F32", shape=[1], data_offsets=[4, 8])}).encode()
    header = struct.pack("<Q", len(raw)) + raw
    data = header + b"abcdefgh"
    segments = [dict(key=key, target_range=[len(header) + i * 4, len(header) + i * 4 + 3],
        source_range=[200 + i * 4, 203 + i * 4], source_file_bytes=1000,
        url="https://huggingface.co/Qwen/fixture/resolve/" + "1" * 40 + "/model.safetensors",
        sha256=hashlib.sha256(data[len(header) + i * 4:len(header) + i * 4 + 4]).hexdigest())
        for i, key in enumerate(("a", "b"))]
    manifest = dict(header_base64=base64.b64encode(header).decode(), bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(), segments=segments)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    calls = []

    def fetch(url, lo, hi):
        calls.append((lo, hi))
        return b"abcdefgh"[lo - 200:hi - 199], 1000

    monkeypatch.setattr(module, "fetch_range", fetch)
    return module, path, tmp_path / "model.safetensors", manifest, header, data, calls


def test_exact_rebuild_existing_verification_and_refuse_overwrite(setup):
    module, path, output, _, _, data, calls = setup
    module.rebuild(path, output)
    assert output.read_bytes() == data and len(calls) == 2
    module.rebuild(path, output)
    assert len(calls) == 2
    output.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="refuse to replace"):
        module.rebuild(path, output)


def test_resume_checks_written_segment_content_not_just_a_journal(setup):
    module, path, output, _, header, data, calls = setup
    output.with_suffix(".safetensors.building").write_bytes(header + b"abcd" + b"\0" * 4)
    module.rebuild(path, output)
    assert calls == [(204, 207)] and output.read_bytes() == data


@pytest.mark.parametrize("mutation", ["unpin", "overlap", "size", "missing", "badsha"])
def test_invalid_manifest_or_payload_never_publishes_final(setup, mutation):
    module, path, output, manifest, _, _, _ = setup
    if mutation == "unpin":
        manifest["segments"][0]["url"] = "https://huggingface.co/Qwen/fixture/resolve/main/model.safetensors"
    elif mutation == "overlap":
        manifest["segments"][1]["target_range"][0] -= 1
    elif mutation == "size":
        manifest["bytes"] += 1
    elif mutation == "missing":
        manifest["segments"].pop()
    else:
        manifest["segments"][1]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.rebuild(path, output)
    assert not output.exists()
