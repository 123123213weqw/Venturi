"""Native forward/control/cache loop parity, ownership, source and failure checks."""

import ctypes
import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from test_fp32_parameters import mixed_port as base_port
from test_qwen_model import hybrid_fixture
from venturi.models import NativeQwenModel, ResidentQwenModel, StreamingQwenModel
from venturi.models.qwen_native import Config, Span, Matrix, Layer
from venturi.models.qwen_rope import QwenTextRoPE
from venturi.ops import TorchBackend


def test_native_descriptor_layout_and_explicit_backend():
    assert [ctypes.sizeof(t) for t in (Config, Span, Matrix, Layer)] == [
        96,
        40,
        24,
        264,
    ]
    with pytest.raises(TypeError, match="explicit CudaBackend"):
        NativeQwenModel.from_q8(None, TorchBackend(), weight_mode="streaming")


def test_rope_inverse_initialization_is_cpu_even_under_global_factory_context(tmp_path):
    weights, _ = hybrid_fixture(tmp_path)
    with torch.device("meta"):
        rope = QwenTextRoPE.from_config(weights.config)
    assert rope.inverse_frequency.device.type == "cpu"


@pytest.fixture(params=["native-fp16", "native-bf16"])
def native_engine(request):
    return base_port.__wrapped__(request)


def equal_snapshot(actual, reference, position):
    for kind, slots in (("gdn", reference._states), ("kv", reference._caches)):
        assert len(actual[kind]) == len(slots)
        for a, b in zip(actual[kind], slots):
            assert (a is None) == (b is None)
            if a is not None:
                expected = (
                    (b.history, b.recurrent) if kind == "gdn" else (b.keys, b.values)
                )
                for x, y in zip(a, expected):
                    assert torch.equal(x, y.cpu()), (
                        kind,
                        position,
                        float((x.double() - y.cpu().double()).abs().max()),
                    )


@pytest.mark.parametrize("mode", ["streaming", "resident"])
def test_native_complete_hybrid_logits_state_trace_and_chunk_continuation(
    tmp_path, native_engine, mode
):
    b, dtype, dev = native_engine
    weights, _ = hybrid_fixture(tmp_path)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode=mode,
        dtype=dtype,
        device=dev,
        max_tokens=3,
        reserve_bytes=0,
    )
    reference = ResidentQwenModel.from_q8(
        weights, b, dtype=dtype, device=dev, reserve_bytes=0
    ).create_session(20, max_tokens=3, reserve_bytes=0)
    s = model.create_session(20)
    # Snapshot before any input must support empty KV prefixes.
    assert all(kv is None or kv[0].numel() == 0 for kv in s.snapshot()["kv"])
    uploads = s.info()["model_weight_upload_bytes"]
    for ids in ([1, 2, 3, 4, 5], [6], [7, 8]):
        trace = []
        actual = s.forward(ids, all_logits=True, trace=trace)
        expected = reference.forward(ids, all_logits=True).cpu()
        assert torch.equal(actual, expected), float((actual - expected).abs().max())
        assert s.position == reference.position and not s.failed
        assert len(trace) == ((len(ids) + 2) // 3) * 4
        equal_snapshot(s.snapshot(), reference, s.position)
    if mode == "resident":
        assert s.info()["model_weight_upload_bytes"] == uploads
    else:
        # Selected embedding rows, not a complete vocabulary upload per step.
        h = (model.config["hidden_size"] + 127) // 128 * 128
        assert s.info()["streaming_embedding_upload_bytes"] == 8 * (h + h // 32 * 2)
    saved = actual.clone()
    s.forward([1])
    assert torch.equal(actual, saved)  # Returned CPU logits own their storage.
    s.close()
    reference.close()
    model.close()


@pytest.mark.parametrize("mode", ["streaming", "resident"])
def test_generation_entirely_native_no_python_model_or_torch_math(
    tmp_path, native_engine, mode, monkeypatch
):
    b, dtype, dev = native_engine
    weights, _ = hybrid_fixture(tmp_path)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode=mode,
        dtype=dtype,
        device=dev,
        max_tokens=3,
        reserve_bytes=0,
    )
    reference = ResidentQwenModel.from_q8(
        weights, b, dtype=dtype, device=dev, reserve_bytes=0
    ).create_session(20, max_tokens=3, reserve_bytes=0)
    prompt, generated = [1, 4, 3, 2, 1], []
    for step in range(4):
        logits = reference.forward(prompt if step == 0 else [generated[-1]])
        generated.append(int(logits.cpu().argmax()))
    s = model.create_session(20)

    def forbidden(*a, **k):
        raise AssertionError(
            "Native generation cannot call Python forward/operator/math"
        )

    with monkeypatch.context() as patch:
        patch.setattr(StreamingQwenModel, "create_session", forbidden)
        patch.setattr(ResidentQwenModel, "create_session", forbidden)
        patch.setattr(weights.__class__, "read_linear", forbidden)
        for name in ("matmul", "mm", "bmm", "argmax", "cos", "sin", "empty", "zeros"):
            patch.setattr(torch, name, forbidden)
        for name in ("matmul", "__matmul__", "argmax", "float", "to", "cpu"):
            patch.setattr(torch.Tensor, name, forbidden)
        assert s.generate(prompt, max_new_tokens=4) == generated
    assert s.position == reference.position == len(prompt) + 3
    equal_snapshot(s.snapshot(), reference, s.position)
    s.reset()
    assert (
        s.generate(prompt, max_new_tokens=4, eos_token_ids=[generated[0]])
        == generated[:1]
    )
    assert s.position == len(prompt)
    s.close()
    reference.close()
    model.close()


def test_native_failure_atomic_cursor_reset_and_independent_requests(
    tmp_path, native_engine
):
    b, dtype, dev = native_engine
    weights, _ = hybrid_fixture(tmp_path)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode="streaming",
        dtype=dtype,
        device=dev,
        max_tokens=3,
        reserve_bytes=0,
    )
    a, c = model.create_session(20), model.create_session(20)
    clean = c.forward([1, 2, 3])
    a._inject_failure(layer=2, position=3)
    with pytest.raises(RuntimeError, match="injected"):
        a.forward([1, 2, 3, 4, 5, 6])
    assert a.failed and a.position == 0 and c.position == 3
    with pytest.raises(RuntimeError, match="reset required"):
        a.generate([1], max_new_tokens=1)
    a.reset()
    assert not a.failed and a.position == 0 and torch.equal(a.forward([1, 2, 3]), clean)
    before = a.snapshot()
    for ids in ([-1], [True], [model.config["vocab_size"]]):
        with pytest.raises(ValueError):
            a.forward(ids)
    assert a.position == 3 and not a.failed
    assert torch.equal(before["gdn"][0][1], a.snapshot()["gdn"][0][1])
    a.close()
    c.close()
    model.close()


def test_native_owned_constants_model_close_and_metadata_isolation(
    tmp_path, native_engine
):
    b, dtype, dev = native_engine
    weights, _ = hybrid_fixture(tmp_path)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode="resident",
        dtype=dtype,
        device=dev,
        max_tokens=3,
        reserve_bytes=0,
    )
    s = model.create_session(12)
    expected = s.forward([1, 2, 3])
    s.reset()
    weights.artifact.manifest["bindings"].clear()  # Caller metadata is not shared.
    cfg = model.config
    cfg["hidden_size"] = 1  # Public config is a detached copy.
    gc.collect()
    model.close()
    model.close()
    assert torch.equal(s.forward([1, 2, 3]), expected)
    with pytest.raises(RuntimeError, match="model closed"):
        model.create_session(12)
    s.close()
    s.close()
    with pytest.raises(RuntimeError, match="session closed"):
        s.forward([1])


def test_native_source_change_preflight_and_capacity_guards(tmp_path, native_engine):
    b, dtype, dev = native_engine
    weights, _ = hybrid_fixture(tmp_path)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode="streaming",
        dtype=dtype,
        device=dev,
        max_tokens=3,
        reserve_bytes=0,
    )
    s = model.create_session(8)
    with pytest.raises(ValueError, match="capacity"):
        s.generate([1] * 7, max_new_tokens=2)
    assert s.position == 0 and not s.failed
    weights.artifact.path.touch()
    with pytest.raises(RuntimeError, match="source changed"):
        s.forward([1])
    ids = (ctypes.c_int64 * 1)(1)
    out = (ctypes.c_float * model.config["vocab_size"])()
    with pytest.raises(RuntimeError, match="source changed"):
        b._check(
            model._native.venturi_qwen_forward(
                s._handle, ids, 1, out, len(out), 0, None, 0
            )
        )
    assert s.position == 0 and not s.failed
    s.close()
    model.close()


def test_serialized_concurrent_native_requests_keep_independent_state(
    tmp_path, native_engine
):
    b, dtype, dev = native_engine
    weights, _ = hybrid_fixture(tmp_path)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode="resident",
        dtype=dtype,
        device=dev,
        max_tokens=3,
        reserve_bytes=0,
    )
    sessions = [model.create_session(16) for _ in range(4)]
    prompts = [[1, i + 2, i + 3] for i in range(4)]
    expected = [s.forward(p) for s, p in zip(sessions, prompts)]
    for s in sessions:
        s.reset()
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(lambda pair: pair[0].forward(pair[1]), zip(sessions, prompts))
        )
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(s.position == 3 and not s.failed for s in sessions)
    for s in sessions:
        s.close()
    model.close()


def test_cpu_output_budget_default_factory_and_resident_admission(
    tmp_path, native_engine, monkeypatch
):
    b, dtype, dev = native_engine
    weights, _ = hybrid_fixture(tmp_path)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode="streaming",
        dtype=dtype,
        device=dev,
        max_tokens=3,
        reserve_bytes=0,
    )
    s = model.create_session(12)
    with pytest.raises(MemoryError, match="diagnostic"):
        s.forward([1], max_cpu_output_bytes=1)
    assert s.position == 0 and not s.failed
    with torch.device("meta"):
        out = s.forward([1, 2, 3])
        state = s.snapshot()
    assert out.device.type == "cpu" and state["gdn"][0][1].device.type == "cpu"
    s.close()
    model.close()

    def forbidden(*a, **k):
        raise AssertionError("Admission before payload access")

    with monkeypatch.context() as patch:
        patch.setattr(weights.__class__, "read_float", forbidden)
        with pytest.raises(MemoryError, match="No implicit offload"):
            NativeQwenModel.from_q8(
                weights,
                b,
                weight_mode="resident",
                dtype=dtype,
                device=dev,
                reserve_bytes=2**64 - 1,
            )
