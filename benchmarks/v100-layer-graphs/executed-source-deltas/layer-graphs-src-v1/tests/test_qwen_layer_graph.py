"""Target-layer graphs must preserve every committed target/draft state."""

import os
from types import SimpleNamespace

import pytest
import torch
from test_qwen_mtp import mtp_fixture, load_models
from test_qwen_tensor_parallel import same_state
import test_qwen_mtp2 as prior
from venturi.models import NativeQwenMTPSession
from venturi.ops import CudaBackend


@pytest.fixture
def engine():
    path = os.environ.get("VENTURI_LAYER_GRAPH_LIBRARY")
    if not path:
        pytest.skip("explicit target-layer graph library required")
    return CudaBackend(path), torch.float16, "cuda:0"


@pytest.mark.parametrize(
    "forced", ["native", "accept", "reject-first", "reject-second"]
)
def test_graph_exact_recurrent_kv_prefix_eos_and_reset(
    tmp_path, engine, forced, monkeypatch
):
    original = NativeQwenMTPSession.generate
    observed = []

    def generate(self, *args, **kw):
        graph = kw.get("draft_tokens", 1) == 2
        if graph:
            kw["recovery"] = "prefix"
            if not getattr(self, "_target_graph_enabled", False):
                self.enable_target_layer_graph()
        ids = original(self, *args, **kw)
        if graph:
            info = self.target_layer_graph_info()
            observed.append(info)
            assert (
                info["enabled"]
                and info["device_position_bytes"] == info["pinned_position_bytes"] == 4
            )
        return ids

    monkeypatch.setattr(NativeQwenMTPSession, "generate", generate)
    prior.test_depth2_output_target_and_corrected_draft_caches(
        tmp_path, engine, "tp", "q8-residual", forced
    )
    assert observed and any(
        r["replays"] > 1 and r["ready"] and r["nodes"] > 0 for r in observed
    )
    assert max(r["captures"] for r in observed) == 1


def test_graph_failure_after_capture_reset_and_closed_owner(tmp_path, engine):
    a, b, _, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, engine, "tp", profile="q8-residual")
    ref, s = target.create_session(40), model.create_session(40)
    try:
        prompt = [1, 2, 3, 4, 5]
        expected = ref.generate(prompt, max_new_tokens=12)
        s.enable_target_layer_graph()
        s._force_proposals([(v + 1) % a.config["vocab_size"] for v in expected])
        s._inject_failure(layer=0, position=7)
        with pytest.raises(RuntimeError, match="injected"):
            s.generate(prompt, max_new_tokens=12, draft_tokens=2, recovery="prefix")
        assert s.failed and s.target_layer_graph_info()["replays"] > 0
        s.reset()
        assert not s.target_layer_graph_info()["ready"]
        s._force_proposals([])
        ref.reset()
        expected = ref.generate(prompt, max_new_tokens=12)
        target.close()
        model.close()
        assert (
            s.generate(prompt, max_new_tokens=12, draft_tokens=2, recovery="prefix")
            == expected
        )
        same_state(s.snapshot(), ref.snapshot())
        assert s.target_layer_graph_info()["captures"] == 2
    finally:
        ref.close()
        s.close()
        model.close()
        target.close()


def test_graph_configuration_native_and_python_preflight(tmp_path, engine):
    a, b, _, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, engine, "tp", profile="fp16")
    s = model.create_session(20)
    try:
        with pytest.raises(RuntimeError, match="residual"):
            s.enable_target_layer_graph()
        assert not s.target_layer_graph_info()["enabled"]
        fn = model._native.venturi_qwen_layer_graph
        assert fn(s._handle, 2) == -1
        assert fn(None, 1) == -1
    finally:
        s.close()
        model.close()
        target.close()


def test_graph_sdk_preflight_and_older_library():
    s = NativeQwenMTPSession.__new__(NativeQwenMTPSession)
    s.model = SimpleNamespace(_native=SimpleNamespace())
    with pytest.raises(ValueError, match="boolean"):
        s.enable_target_layer_graph(1)
    with pytest.raises(RuntimeError, match="export absent"):
        s.enable_target_layer_graph()
    s._target_graph_enabled = True
    with pytest.raises(ValueError, match="MTP2 prefix"):
        s.generate([1], max_new_tokens=2)
