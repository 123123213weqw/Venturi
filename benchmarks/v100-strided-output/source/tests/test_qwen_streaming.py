"""Explicit streaming mode must match resident math without retaining weights."""

import gc
import weakref

import pytest
import torch

from test_fp32_parameters import mixed_port as base_port
from test_qwen_model import hybrid_fixture, independent_forward
from venturi.models import (
    StreamingQwenModel,
    ResidentQwenModel,
    ResidentQwenGDNLayer,
    ResidentQwenAttentionLayer,
    qwen_request_bytes,
)
from venturi.ops import TorchBackend


@pytest.fixture(params=["cpu-fp16", "native-fp16", "cpu-bf16", "native-bf16"])
def stream_engine(request):
    return base_port.__wrapped__(request)


def models(weights, engine):
    b, d, dev = engine
    return (
        ResidentQwenModel.from_q8(weights, b, dtype=d, device=dev, reserve_bytes=0),
        StreamingQwenModel.from_q8(weights, b, dtype=d, device=dev, reserve_bytes=0),
    )


def test_fp64_full_source_oracle_and_prefix_continuation(tmp_path):
    weights, raw = hybrid_fixture(tmp_path)
    _, model = models(weights, (TorchBackend(), torch.float64, "cpu"))
    ids = [2, 4, 1, 8, 3, 9, 2]
    ideal = independent_forward(raw, model.config, ids, tied=False)
    events = []
    s = model.create_session(16, max_tokens=3, observer=events.append)
    trace = []
    actual = s.forward(ids, all_logits=True, trace=trace)
    torch.testing.assert_close(actual, ideal, atol=3e-11, rtol=3e-10)
    assert s.position == 7 and not s.failed and len(trace) == 12
    assert all(t["hidden"].device.type == "cpu" for t in trace)
    assert sum(e["phase"] == "layer" for e in events) == 12
    s.reset()
    split = torch.cat(
        (
            s.forward(ids[:2], all_logits=True),
            s.forward(ids[2:5], all_logits=True),
            s.forward(ids[5:], all_logits=True),
        )
    )
    torch.testing.assert_close(split, ideal, atol=3e-11, rtol=3e-10)
    assert s.cache_bytes == qwen_request_bytes(model.config, 16, dtype=torch.float64)
    s.close()


def test_resident_streaming_logits_state_and_cache_parity(tmp_path, stream_engine):
    weights, _ = hybrid_fixture(tmp_path)
    resident, stream = models(weights, stream_engine)
    a, b = (
        resident.create_session(20, max_tokens=3, reserve_bytes=0),
        stream.create_session(20, max_tokens=3),
    )
    for ids in ([1, 2, 3, 4, 5], [6], [7, 8]):
        expected = a.forward(ids, all_logits=True)
        actual = b.forward(ids, all_logits=True)
        assert torch.equal(actual, expected) and a.position == b.position
        for r, s in zip(a._states, b._states):
            if r is not None:
                assert torch.equal(r.history, s.history) and torch.equal(
                    r.recurrent, s.recurrent
                )
        for r, s in zip(a._caches, b._kv):
            if r is not None:
                assert torch.equal(r.keys, s.keys[: b.position]) and torch.equal(
                    r.values, s.values[: b.position]
                )
    assert set(b._lanes) == {1, 3} and b.workspace_bytes > 0
    assert (
        b.cache_bytes
        == a.cache_bytes
        == qwen_request_bytes(stream.config, 20, dtype=stream.dtype)
    )
    a.close()
    b.close()


def test_no_resident_layer_retention_and_owned_logits(
    tmp_path, stream_engine, monkeypatch
):
    weights, _ = hybrid_fixture(tmp_path)
    _, m = models(weights, stream_engine)
    seen = []
    for cls in (ResidentQwenGDNLayer, ResidentQwenAttentionLayer):
        original = cls.from_q8.__func__

        def wrap(cls, *args, _original=original, **kwargs):
            r = _original(cls, *args, **kwargs)
            seen.append(weakref.ref(r))
            return r

        monkeypatch.setattr(cls, "from_q8", classmethod(wrap))
    s = m.create_session(16, max_tokens=3)
    x = s.forward([1, 2, 3], all_logits=True)
    copy = x.clone()
    gc.collect()
    assert seen and all(r() is None for r in seen)
    assert m.persistent_weight_bytes < m.peak_phase_weight_bytes
    s.forward([4, 5])
    gc.collect()
    assert torch.equal(copy, x) and all(r() is None for r in seen)
    s.close()


def test_streaming_poison_atomic_position_reset_and_independent_request(
    tmp_path, stream_engine
):
    weights, _ = hybrid_fixture(tmp_path)
    _, m = models(weights, stream_engine)

    def fail(event):
        if event["phase"] == "layer" and event["layer"] == 3 and event["position"] == 3:
            raise RuntimeError("injected second-chunk observer failure")

    a = m.create_session(16, max_tokens=3, observer=fail)
    b = m.create_session(16, max_tokens=3)
    clean = b.forward([1, 2, 3], all_logits=True)
    with pytest.raises(RuntimeError, match="second-chunk"):
        a.forward([1, 2, 3, 4, 5, 6])
    assert a.failed and a.position == 0 and b.position == 3
    with pytest.raises(RuntimeError, match="reset required"):
        a.forward([1])
    a.reset()
    a._observer = None
    assert torch.equal(a.forward([1, 2, 3], all_logits=True), clean)
    a.close()
    b.close()


def test_manifest_isolation_change_detection_and_metadata_errors(
    tmp_path, stream_engine
):
    weights, _ = hybrid_fixture(tmp_path)
    _, m = models(weights, stream_engine)
    weights.artifact.manifest["components"]["text"]["config"]["hidden_size"] = 1
    s = m.create_session(16, max_tokens=3)
    with pytest.raises(ValueError, match="Vocabulary"):
        s.forward([True])
    with pytest.raises(ValueError, match="capacity"):
        s.forward([1] * 17)
    assert s.position == 0 and not s.failed and not s._lanes
    s.forward([1])
    m.config["rms_norm_eps"] *= 2
    with pytest.raises(RuntimeError, match="config/metadata"):
        s.forward([1])
    assert s.position == 1 and not s.failed
    s.close()


def test_streaming_prefix_import_metadata_alias_and_lifecycle(tmp_path, stream_engine):
    weights, _ = hybrid_fixture(tmp_path)
    r, _ = models(weights, stream_engine)
    layer = r.layers[1]
    cache = layer.empty_cache(capacity=6)
    keys = torch.randn(3, 1, 64, dtype=r.dtype, device=r.device)
    values = torch.randn_like(keys)
    with pytest.raises(ValueError, match="alias"):
        cache.load_prefix(cache._keys[:3], values)
    with pytest.raises(ValueError, match="shape"):
        cache.load_prefix(keys, values[:2])
    assert cache.length == 0 and not cache.failed
    cache.load_prefix(keys, values)
    assert (
        cache.length == 3
        and torch.equal(cache.keys, keys)
        and torch.equal(cache.values, values)
    )
    with pytest.raises(ValueError, match="empty cache"):
        cache.load_prefix(keys, values)
    cache.reset()
    cache._failed = True
    with pytest.raises(RuntimeError, match="reset required"):
        cache.load_prefix(keys, values)


def test_streaming_source_file_change_never_commits(tmp_path):
    weights, _ = hybrid_fixture(tmp_path)
    m = StreamingQwenModel.from_q8(
        weights, TorchBackend(), device="cpu", dtype=torch.float64, reserve_bytes=0
    )
    s = m.create_session(16, max_tokens=3)
    # Same-length file change must still be detected (including mtime/ctime).
    path = weights.artifact.path
    with path.open("r+b") as f:
        f.seek(-1, 2)
        byte = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([byte[0] ^ 1]))
    with pytest.raises(RuntimeError, match="artifact changed"):
        s.forward([1])
    assert not s.failed and s.position == 0
    s.close()


def test_streaming_wrong_stream_rejected(tmp_path, stream_engine):
    if stream_engine[2] == "cpu":
        pytest.skip("CUDA stream ownership")
    weights, _ = hybrid_fixture(tmp_path)
    _, m = models(weights, stream_engine)
    owner, other = (
        torch.cuda.Stream(device=m.device),
        torch.cuda.Stream(device=m.device),
    )
    with torch.cuda.stream(owner):
        s = m.create_session(16, max_tokens=3)
        expected = s.forward([1])
    with torch.cuda.stream(other):
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            s.forward([2])
    assert not s.failed and s.position == 1
    with torch.cuda.stream(owner):
        s.close()
    assert bool(expected.isfinite().all())


def test_streaming_native_has_no_matrix_attention_fallback(
    tmp_path, stream_engine, monkeypatch
):
    if stream_engine[2] == "cpu":
        pytest.skip("native-only no-fallback assertion")
    weights, _ = hybrid_fixture(tmp_path)
    _, m = models(weights, stream_engine)
    s = m.create_session(16, max_tokens=3)

    def forbidden(*a, **k):
        raise AssertionError(
            "native streaming used a numerical matrix/attention Torch fallback"
        )

    with monkeypatch.context() as patch:
        for name in ("matmul", "mm", "bmm"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(torch.Tensor, "__matmul__", forbidden)
        patch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
        output = s.forward([1, 2, 3])
    assert bool(output.isfinite().all())
    s.close()
