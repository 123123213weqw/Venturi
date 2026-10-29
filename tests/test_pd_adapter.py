"""CPU-only Python PD preflight; these stubs are not native GPU validation."""
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest
import torch

from venturi.models.qwen_batch import NativeQwenBatchSession
from venturi.models.qwen_native import NativeQwenModel
from venturi.models.qwen_pd import NativeQwenPDWorker, NativeQwenStateTransport


class Function:
    def __init__(self):
        self.calls = []
        self.state = self.destination = None

    def __call__(self, *args):
        self.calls.append(args[:-1])
        output = args[-1]
        output[:] = [self.state.nbytes, self.state.position, len(self.state.components), 0]
        self.destination.position = self.state.position
        return 0


def fake_worker(library, handle):
    model = NativeQwenModel()
    model._mode, model._tensor_parallel_device, model._kv_cache_mode = "resident", None, "static"
    model._metadata = lambda: None
    model._backend = SimpleNamespace(_native=library, _check=lambda status: None)
    model._digest, model._dtype, model._projection_profile = b"x" * 32, torch.float16, "fp16"
    model._max_tokens = 8
    model._cfg = dict(vocab_size=17, linear_num_key_heads=2, linear_key_head_dim=64,
        linear_num_value_heads=4, linear_value_head_dim=13, linear_conv_kernel_dim=4,
        num_key_value_heads=1, head_dim=64, layer_types=["linear_attention", "full_attention"])
    batch = NativeQwenBatchSession()
    batch.model, batch.capacities = model, (8,)
    batch._lock, batch._handle = threading.RLock(), handle
    batch.position = 0
    batch._check = lambda: None
    batch.info = lambda: {"requests": [dict(position=batch.position, failed=False, capacity=8)]}
    def reset(*, slots):
        batch.position = 0
    def step(prompts, **kwargs):
        batch.position += len(prompts[0])
        return [1]
    batch.reset, batch.step = reset, step
    return NativeQwenPDWorker(batch)


def test_native_pd_python_descriptors_epoch_and_foreign_library_preflight():
    function = Function()
    library = SimpleNamespace(_handle=17, venturi_qwen_batch_handoff=function)
    p, d = fake_worker(library, 10), fake_worker(library, 20)
    transport = NativeQwenStateTransport(mode="peer")
    p.step([[1, 2]], slots=[0], sample_slots=[0])
    state = p.export_state(0, position=2)
    assert state.position == 2 and len(state.components) == 4
    assert state.components[1].shape == (4, 13, 64)
    assert state.components[1].dtype == "float32"
    function.state, function.destination = state, d.batch
    receipt = transport.transfer(state, d, slot=0)
    assert receipt.payload_bytes == state.nbytes and receipt.position == 2
    assert function.calls[-1] == (10, 0, 20, 0, 2, 1)
    with pytest.raises(ValueError, match="fresh"):
        transport.transfer(state, d, slot=0)
    p.reset(slots=[0])
    d.reset(slots=[0])
    with pytest.raises(ValueError, match="stale"):
        transport.transfer(state, d, slot=0)
    p.step([[1]], slots=[0], sample_slots=[0])
    state = p.export_state(0, position=1)
    d.state_identity = replace(d.state_identity, arithmetic="different")
    with pytest.raises(ValueError, match="identity mismatch"):
        transport.transfer(state, d, slot=0)
    d.state_identity = p.state_identity
    d.batch.model._backend = SimpleNamespace(_native=SimpleNamespace(_handle=99))
    with pytest.raises(ValueError, match="same loaded native library"):
        transport.transfer(state, d, slot=0)
    assert len(function.calls) == 1


def test_native_pd_refuses_unsupported_targets_and_implicit_transport():
    with pytest.raises(TypeError, match="ordinary native"):
        NativeQwenPDWorker(object())
    library = SimpleNamespace(_handle=17, venturi_qwen_batch_handoff=Function())
    worker = fake_worker(library, 10)
    for name, value in (("_tensor_parallel_device", torch.device("cuda:1")),
                        ("_kv_cache_mode", "vram-first"), ("_mode", "streaming")):
        old = getattr(worker.batch.model, name)
        setattr(worker.batch.model, name, value)
        with pytest.raises(ValueError, match="resident TP1"):
            NativeQwenPDWorker(worker.batch)
        setattr(worker.batch.model, name, old)
    for mode in ("auto", "tcp", None):
        with pytest.raises(ValueError):
            NativeQwenStateTransport(mode=mode)
