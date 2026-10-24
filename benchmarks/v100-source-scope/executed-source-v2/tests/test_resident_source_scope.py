"""Execution-only source-check deduplication must retain owner/boundary guards."""

from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from test_qwen_model import hybrid_fixture
from test_qwen_mtp import load_models, mtp_fixture
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenModel
from venturi.ops import CudaBackend
from venturi.ops.build import build_blocks_backend


@pytest.mark.parametrize("value", [1, "true", None])
def test_build_policy_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        build_blocks_backend("unused", architecture="sm70", resident_source_scope=value)


def test_build_policy_rejects_other_architecture_before_compiler():
    with pytest.raises(ValueError, match="SM70 blocks"):
        build_blocks_backend("unused", architecture="sm89", resident_source_scope=True)


def test_build_policy_is_opt_in_and_cache_separated(tmp_path, monkeypatch):
    commands = []

    def compiler(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "fake compiler version", "")
        commands.append(command)
        Path(command[-1]).write_bytes(b"Non-executable fake compiler output")
        return subprocess.CompletedProcess(command, 0, "fake build\n", "")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", compiler)
    original = build_blocks_backend(tmp_path, nvcc="fake-nvcc", architecture="sm70")
    scoped = build_blocks_backend(
        tmp_path, nvcc="fake-nvcc", architecture="sm70", resident_source_scope=True
    )
    assert original != scoped
    assert not any("RESIDENT_SOURCE_SCOPE" in s for s in commands[0])
    assert "-DVENTURI_RESIDENT_SOURCE_SCOPE=1" in commands[1]
    assert not json.loads(original.with_name("build.json").read_text())[
        "resident_source_scope"
    ]
    assert json.loads(scoped.with_name("build.json").read_text())[
        "resident_source_scope"
    ]
    assert (
        build_blocks_backend(
            tmp_path, nvcc="fake-nvcc", architecture="sm70", resident_source_scope=True
        )
        == scoped
    )
    assert len(commands) == 2


def test_metadata_reports_policy_without_changing_arithmetic_profile():
    m = NativeQwenModel()
    m._mode, m._projection_profile = "resident", "q8-residual-column-mlp"
    m._backend = SimpleNamespace(
        _native=SimpleNamespace(venturi_qwen_source_scope_mode=lambda: 1)
    )
    assert m.projection_info["source_validation_policy"].startswith("resident-forward")
    assert m.projection_info["projection_profile"] == "q8-residual-column-mlp"
    m._mode = "streaming"
    assert m.projection_info["source_validation_policy"] == "per-weight"
    m._backend = SimpleNamespace(_native=SimpleNamespace())
    assert m.projection_info["source_validation_policy"] == "per-weight"


@pytest.mark.parametrize("mode", ["resident", "streaming", "column"])
def test_default_build_still_matches_qualified_control(tmp_path, mode):
    paths = [
        os.environ.get(name)
        for name in (
            "VENTURI_SOURCE_SCOPE_DEFAULT_LIBRARY",
            "VENTURI_SOURCE_SCOPE_CONTROL_LIBRARY",
        )
    ]
    if not all(paths):
        pytest.skip(
            "Explicit fresh default and qualified per-weight libraries required"
        )
    weights, _ = hybrid_fixture(tmp_path)
    default_backend = CudaBackend(paths[0])
    assert default_backend._native.venturi_qwen_source_scope_mode() == 0
    first, control = (
        load(weights, default_backend, mode),
        load(weights, CudaBackend(paths[1]), mode),
    )
    sessions = [first.create_session(32), control.create_session(32)]
    try:
        for ids in ([1, 2, 3, 4], [5]):
            assert torch.equal(*[s.forward(ids, all_logits=True) for s in sessions])
            same_state(*[s.snapshot() for s in sessions])
        assert first.projection_info["source_validation_policy"] == "per-weight"
    finally:
        for s in sessions:
            s.close()
        first.close()
        control.close()


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_SOURCE_SCOPE_LIBRARY")
    if not path:
        pytest.skip("Explicit experimental resident-source-scope library required")
    b = CudaBackend(path)
    n = b._native
    n.venturi_qwen_source_scope_mode.argtypes = []
    n.venturi_qwen_source_scope_mode.restype = C.c_int
    assert n.venturi_qwen_source_scope_mode() == 1
    for name, args in (
        ("venturi_qwen_debug_source_pause", [C.c_void_p, C.c_int]),
        ("venturi_qwen_mtp_debug_source_pause", [C.c_void_p, C.c_int]),
        ("venturi_qwen_debug_source_scopes", [C.c_void_p, C.c_void_p]),
    ):
        fn = getattr(n, name)
        fn.argtypes, fn.restype = args, C.c_int
    return b


def load(weights, backend, mode="resident"):
    kwargs = dict(
        weight_mode="streaming" if mode == "streaming" else "resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=3,
        reserve_bytes=0,
    )
    if mode == "column":
        kwargs.update(
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            projection_profile="q8-residual-column-mlp",
        )
    return NativeQwenModel.from_q8(weights, backend, **kwargs)


def mutate(path, kind):
    if kind == "touch":
        old = path.stat()
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 123456789))
    else:
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)


def paused_call(b, fn, handle, call, mutation):
    """Pause after GPU completion but before the mandatory native exit check."""
    b._check(fn(handle, 1))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call)
        try:
            deadline = time.monotonic() + 20
            while fn(handle, -1) != 2:
                if future.done():
                    future.result()
                    pytest.fail("Native call returned before its armed boundary")
                assert time.monotonic() < deadline, (
                    "Verified native boundary did not become ready"
                )
                time.sleep(0.001)
            mutation()
        finally:
            b._check(fn(handle, 0))
        with pytest.raises(RuntimeError, match="source changed"):
            future.result(timeout=20)


@pytest.mark.parametrize("mode", ["resident", "streaming", "column"])
@pytest.mark.parametrize("mutation", ["touch", "replace"])
def test_mid_forward_mutation_rejected_at_native_exit(
    tmp_path, backend, mode, mutation
):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, backend, mode)
    s = model.create_session(32)
    try:
        paused_call(
            backend,
            backend._native.venturi_qwen_debug_source_pause,
            model._handle,
            lambda: s.forward([1, 2, 3], all_logits=True),
            lambda: mutate(weights.artifact.path, mutation),
        )
        assert s.failed and s.position == 0
        with pytest.raises(RuntimeError, match="source changed"):
            s.reset()
    finally:
        s.close()
        model.close()
    # Changed stamp requires recreation, not silently accepting it on reset.
    model = load(weights, backend, mode)
    s = model.create_session(32)
    try:
        assert torch.isfinite(s.forward([1, 2])).all()
    finally:
        s.close()
        model.close()


@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_nested_foreign_and_exception_scope_ownership(tmp_path, backend, mode):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    wa, _ = hybrid_fixture(a)
    wb, _ = hybrid_fixture(b)
    first, other = load(wa, backend), load(wb, backend, mode)
    try:
        fn = backend._native.venturi_qwen_debug_source_scopes
        backend._check(fn(first._handle, other._handle))
        mutate(wb.artifact.path, "touch")
        with pytest.raises(RuntimeError, match="source changed"):
            backend._check(fn(first._handle, other._handle))
        s = first.create_session(32)
        try:
            assert torch.isfinite(s.forward([1, 2])).all()
        finally:
            s.close()
    finally:
        other.close()
        first.close()


@pytest.mark.parametrize("mode", ["resident", "column"])
def test_exact_control_private_threads_failure_reset_and_closed(
    tmp_path, backend, mode
):
    path = os.environ.get("VENTURI_SOURCE_SCOPE_CONTROL_LIBRARY")
    if not path:
        pytest.skip("Qualified per-weight control library required")
    weights, _ = hybrid_fixture(tmp_path)
    first, control = (
        load(weights, backend, mode),
        load(weights, CudaBackend(path), mode),
    )
    sessions, references = (
        [first.create_session(32) for _ in range(2)],
        [control.create_session(32) for _ in range(2)],
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(s.forward, ids, all_logits=True)
                for s, ids in zip(sessions, ([1, 2, 3, 4], [6, 5]))
            ]
            for s, r, ids, future in zip(
                sessions, references, ([1, 2, 3, 4], [6, 5]), futures
            ):
                assert torch.equal(future.result(), r.forward(ids, all_logits=True))
                same_state(s.snapshot(), r.snapshot())
        sessions[0]._inject_failure(layer=2, position=4)
        with pytest.raises(RuntimeError, match="injected"):
            sessions[0].forward([2])
        assert sessions[0].failed
        assert torch.equal(sessions[1].forward([2]), references[1].forward([2]))
        sessions[0].reset()
        references[0].reset()
        assert torch.equal(sessions[0].forward([3, 4]), references[0].forward([3, 4]))
        same_state(sessions[0].snapshot(), references[0].snapshot())
        sessions[0].close()
        with pytest.raises(RuntimeError, match="closed"):
            sessions[0].forward([1])
    finally:
        for s in sessions + references:
            s.close()
        first.close()
        control.close()


@pytest.mark.parametrize("kind", ["target", "draft"])
def test_mid_mtp_mutation_poisoned_without_returned_generation(tmp_path, backend, kind):
    weights, draft, _, _ = mtp_fixture(tmp_path)
    target, model = load_models(
        weights,
        draft,
        (backend, torch.float16, "cuda:0"),
        "tp",
        profile="q8-residual-column-mlp",
        draft_profile="q8-residual",
    )
    s = model.create_session(32)
    s.prepare_prefix()
    try:
        fn = (
            backend._native.venturi_qwen_debug_source_pause
            if kind == "target"
            else backend._native.venturi_qwen_mtp_debug_source_pause
        )
        handle = target._handle if kind == "target" else model._handle
        path = weights.artifact.path if kind == "target" else draft.artifact.path
        paused_call(
            backend,
            fn,
            handle,
            lambda: s.generate(
                [1, 2, 3], max_new_tokens=5, draft_tokens=2, recovery="prefix"
            ),
            lambda: mutate(path, "touch"),
        )
        # MTP's SDK info() also requests stats, which deliberately revalidates
        # metadata. Inspect the native failure bit without refreshing stamps.
        values = (C.c_uint64 * 8)()
        backend._check(model._native.venturi_qwen_info(s._handle, values))
        assert values[1] == 1
        with pytest.raises(RuntimeError, match="source changed"):
            s.info()
    finally:
        s.close()
        model.close()
        target.close()


@pytest.mark.parametrize("mode", ["resident", "column"])
def test_mid_native_batch_mutation_poisoned_cohort(tmp_path, backend, mode):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, backend, mode)
    batch = model.create_batch([32, 32])
    try:
        paused_call(
            backend,
            backend._native.venturi_qwen_debug_source_pause,
            model._handle,
            lambda: batch.generate([[1, 2], [3]], max_new_tokens=4),
            lambda: mutate(weights.artifact.path, "touch"),
        )
        info = batch.info()
        assert all(r["failed"] for r in info["requests"])
    finally:
        batch.close()
        model.close()
