"""Build orchestration tests with a fake compiler; no compilation on the local Mac."""

from pathlib import Path
import subprocess
import sys

import pytest

from venturi.ops.build import build_cuda_backend


def test_explicit_build_cache_and_integrity(monkeypatch, tmp_path):
    calls = []

    def compiler(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "fake compiler version", "")
        calls.append(command)
        Path(command[-1]).write_bytes(b"fake shared object, not executable")
        return subprocess.CompletedProcess(
            command, 0, "complete compiler output\n", "warning\n"
        )

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", compiler)
    library = build_cuda_backend(tmp_path, nvcc="fake-nvcc")
    assert len(calls) == 1
    assert (
        library.with_name("build.log").read_text()
        == "complete compiler output\nwarning\n"
    )
    assert build_cuda_backend(tmp_path, nvcc="fake-nvcc") == library
    assert len(calls) == 1
    library.write_bytes(b"corrupted")
    assert build_cuda_backend(tmp_path, nvcc="fake-nvcc") == library
    assert len(calls) == 2
    library.with_name("build.json").write_text("[]")
    assert build_cuda_backend(tmp_path, nvcc="fake-nvcc") == library
    assert len(calls) == 3


def test_compiler_failure_keeps_full_output(monkeypatch, tmp_path):
    def compiler(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "fake compiler version", "")
        return subprocess.CompletedProcess(command, 1, "stdout\n", "detailed error\n")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", compiler)
    with pytest.raises(RuntimeError, match="CUDA compilation failed"):
        build_cuda_backend(tmp_path, nvcc="fake-nvcc")
    log = next(tmp_path.glob("*/build.log"))
    assert log.read_text() == "stdout\ndetailed error\n"
    assert not list(tmp_path.glob("*/libventuri_ada.so"))


def test_q8_tiles_are_explicit_cached_build_identity(monkeypatch, tmp_path):
    from venturi.ops.build import build_blocks_backend

    commands = []

    def compiler(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "fake compiler", "")
        commands.append(command)
        Path(command[-1]).write_bytes(b"fake library; no compilation")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", compiler)
    a = build_blocks_backend(tmp_path, nvcc="fake", q8_token_tile=1, q8_row_tile=1)
    b = build_blocks_backend(tmp_path, nvcc="fake", q8_token_tile=4, q8_row_tile=4)
    assert a != b
    assert "-DVENTURI_Q8_TOKEN_TILE=1" in commands[0]
    assert "-DVENTURI_Q8_ROW_TILE=4" in commands[1]
    assert (
        build_blocks_backend(tmp_path, nvcc="fake", q8_token_tile=4, q8_row_tile=4) == b
    )
    assert len(commands) == 2
    for bad in (0, 3, True, "4"):
        with pytest.raises(ValueError, match="tiles"):
            build_blocks_backend(tmp_path, q8_token_tile=bad)
    c = build_blocks_backend(
        tmp_path, nvcc="fake", q8_token_tile=4, q8_row_tile=4, q8_decode_mode=2
    )
    assert c != b
    assert "-DVENTURI_Q8_DECODE_MODE=0" in commands[1]
    assert "-DVENTURI_Q8_DECODE_MODE=2" in commands[2]
    assert (
        build_blocks_backend(
            tmp_path, nvcc="fake", q8_token_tile=4, q8_row_tile=4, q8_decode_mode=2
        )
        == c
    )
    assert len(commands) == 3
    for bad in (-1, 3, True, "2"):
        with pytest.raises(ValueError, match="decode mode"):
            build_blocks_backend(tmp_path, q8_decode_mode=bad)


def test_blocks_build_is_separate_and_does_not_compile_external_headers(
    monkeypatch, tmp_path
):
    from venturi.ops.build import build_blocks_backend

    commands = []

    def compiler(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "test nvcc", "")
        commands.append(command)
        Path(command[-1]).write_bytes(b"fake blocks library")
        return subprocess.CompletedProcess(command, 0, "blocks build output", "")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", compiler)
    library = build_blocks_backend(tmp_path, nvcc="fake-nvcc")
    assert library.name == "libventuri_blocks.so"
    assert "-lcublas" in commands[0]
    assert any(arg.endswith("/blocks.cu") for arg in commands[0])
    assert not any("third_party" in arg for arg in commands[0])
    assert build_blocks_backend(tmp_path, nvcc="fake-nvcc") == library
    assert len(commands) == 1


def test_explicit_sm70_target_cache_and_no_ada_retarget(monkeypatch, tmp_path):
    from venturi.ops.build import build_blocks_backend

    commands = []

    def compiler(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "test nvcc 12.9", "")
        commands.append(command)
        Path(command[-1]).write_bytes(b"fake object: no compiler executed")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", compiler)
    a = build_blocks_backend(tmp_path, nvcc="fake-nvcc", architecture="sm70")
    b = build_blocks_backend(tmp_path, nvcc="fake-nvcc", architecture="sm89")
    assert a != b
    assert "-gencode=arch=compute_70,code=sm_70" in commands[0]
    assert "-DVENTURI_BLOCKS_ARCH=70" in commands[0]
    assert "-DVENTURI_BLOCKS_ARCH=89" in commands[1]
    assert "-DVENTURI_Q8_TOKEN_TILE=4" in commands[0]
    assert "-DVENTURI_Q8_ROW_TILE=4" in commands[0]
    assert "-DVENTURI_Q8_TOKEN_TILE=1" in commands[1]
    assert "-DVENTURI_Q8_ROW_TILE=1" in commands[1]
    assert build_blocks_backend(tmp_path, nvcc="fake-nvcc", architecture="sm70") == a
    assert len(commands) == 2
    with pytest.raises(ValueError, match="explicit sm70/sm89"):
        build_blocks_backend(tmp_path, nvcc="fake-nvcc", architecture="sm80")


def test_graph_header_is_a_blocks_build_dependency(monkeypatch, tmp_path):
    import json
    from venturi.ops.build import build_blocks_backend

    def compiler(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "fake compiler", "")
        Path(command[-1]).write_bytes(b"fake library, not compiled")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", compiler)
    library = build_blocks_backend(tmp_path, nvcc="fake", architecture="sm70")
    manifest = json.loads(library.with_name("build.json").read_text())
    assert "qwen_layer_graph.cuh" in manifest["source_sha256"]
    assert len(manifest["source_sha256"]) == 10
