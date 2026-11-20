"""Explicit native CLI routing, pins, resource cleanup and tokenizer-free IDs."""

import json

import pytest

from test_qwen_model import hybrid_fixture
from venturi import cli
from venturi import qwen_native_cli as native


def test_native_metadata_inspect_no_gpu_or_payload_promise(
    tmp_path, capsys, monkeypatch
):
    w, _ = hybrid_fixture(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("Metadata inspection must not create a GPU runtime")

    monkeypatch.setattr(native, "CudaBackend", forbidden)
    assert cli.main(["qwen-native", "inspect", str(w.artifact.path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (
        out["artifact_id"] == w.artifact.artifact_id
        and out["formats"]["q8_g32_fp16"] > 0
    )
    assert out["verified_single_file_sha256"] is None and "not payload" in out["scope"]
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "qwen-native",
                "inspect",
                str(w.artifact.path),
                "--expected-sha256",
                "0" * 64,
            ]
        )
    assert exc.value.code == 1 and "SHA256 mismatch" in capsys.readouterr().err


def test_native_generate_requires_explicit_residency_and_does_not_fallback(
    tmp_path, capsys, monkeypatch
):
    w, _ = hybrid_fixture(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("No backend before complete explicit options")

    monkeypatch.setattr(native, "CudaBackend", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["qwen-native", "generate", str(w.artifact.path), "--token-ids", "1,2,3"]
        )
    assert exc.value.code == 1 and "--weight-mode" in capsys.readouterr().err


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("parallel", [False, True, "pp"])
def test_native_cli_ids_one_generate_call_and_cleanup(
    tmp_path, capsys, monkeypatch, fails, parallel, profile="fp16"
):
    w, _ = hybrid_fixture(tmp_path)
    events = []

    class Session:
        def generate(self, ids, **options):
            events.append(("generate", ids, options))
            if fails:
                raise RuntimeError("native failure, no fallback")
            return [3, 2]

        def info(self):
            return dict(
                position=4, failed=0, model_device_bytes=100, request_device_bytes=200
            )

        def close(self):
            events.append("session-close")

    class Model:
        @classmethod
        def from_q8(cls, weights, backend, **options):
            events.append(("load", options))
            return cls()

        def create_session(self, capacity):
            events.append(("session", capacity))
            return Session()

        def close(self):
            events.append("model-close")

    def no_tokenizer(*a, **k):
        raise AssertionError("Explicit IDs need no tokenizer dependency")

    monkeypatch.setattr(native, "NativeQwenModel", Model)
    monkeypatch.setattr(native, "NativeQwenPipelineModel", Model)
    monkeypatch.setattr(native, "CudaBackend", lambda library: library)
    monkeypatch.setattr(native, "LocalTokenizer", no_tokenizer)
    args = [
        "qwen-native",
        "generate",
        str(w.artifact.path),
        "--token-ids",
        "1,4,3",
        "--blocks-library",
        "explicit.so",
        "--weight-mode",
        "resident" if parallel else "streaming",
        "--dtype",
        "fp16",
        "--max-tokens",
        "3",
        "--max-new-tokens",
        "2",
        "--projection-profile",
        profile,
        "--eos-token-ids",
        "2",
    ]
    if parallel == "pp":
        args += ["--pipeline-device", "cuda:1", "--pipeline-split-layer", "2"]
    elif parallel:
        args += [
            "--tensor-parallel-device",
            "cuda:1",
            "--tensor-parallel-transport",
            "host",
        ]
    if fails:
        with pytest.raises(SystemExit) as exc:
            cli.main(args)
        assert exc.value.code == 1 and "no fallback" in capsys.readouterr().err
    else:
        assert cli.main(args) == 0
        out = json.loads(capsys.readouterr().out)
        assert (
            out["generated_ids"] == [3, 2]
            and out["finish_reason"] == "eos"
            and out["text"] is None
        )
        assert (
            out["weight_mode"] == ("resident" if parallel else "streaming")
            and out["native_memory"]["position"] == 4
        )
        assert not out["whole_model_cuda_graphs"] and "serialized" in out["scheduling"]
    assert events[-2:] == ["session-close", "model-close"]
    calls = [e for e in events if isinstance(e, tuple) and e[0] == "generate"]
    assert len(calls) == 1 and calls[0][1] == [1, 4, 3]
    options = next(e[1] for e in events if isinstance(e, tuple) and e[0] == "load")
    if parallel == "pp":
        assert options["pipeline_device"] == "cuda:1" and options["split_layer"] == 2
        assert (
            options["transport"] == "host" and "tensor_parallel_device" not in options
        )
    else:
        assert options["tensor_parallel_device"] == ("cuda:1" if parallel else None)
        assert options["tensor_parallel_transport"] == ("host" if parallel else "peer")
        assert options["projection_profile"] == profile


def test_residual_profile_cli_is_explicit_and_routes_to_native_target(
    tmp_path, capsys, monkeypatch
):
    test_native_cli_ids_one_generate_call_and_cleanup(
        tmp_path, capsys, monkeypatch, False, True, profile="q8-residual"
    )
