"""Prefill unpack reuse is explicit and only changes qualified single-slot batches."""

from contextlib import closing
import ctypes as C
import os
from types import SimpleNamespace

import pytest
import torch

from qwen_fixtures import source_config
from qwen_safetensors import write_quantized_checkpoint
from venturi.models import NativeQwenModel, QwenWeights
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.models.qwen.qwen_batch import _bind
from venturi.ops import CudaBackend


def test_q5_prefill_reuse_preflight_rejects_multislot_and_wrong_workspace():
    model = NativeQwenModel()
    model._mode, model._max_tokens = "resident", 32
    model._cfg = {"max_position_embeddings": 128}
    with pytest.raises(ValueError, match="explicit bool"):
        model.create_batch([128], q5_prefill_reuse=1)
    with pytest.raises(ValueError, match="one slot"):
        model.create_batch([128] * 4, q5_prefill_reuse=True)
    model._max_tokens = 16
    with pytest.raises(ValueError, match="max_tokens=32"):
        model.create_batch([128], q5_prefill_reuse=True)


def test_q5_prefill_reuse_binding_keeps_old_default_and_requires_additive_symbol():
    def noop(*args):
        return None

    native = SimpleNamespace(
        **{
            "venturi_qwen_batch_" + k: (lambda: None)
            for k in (
                "create",
                "destroy",
                "forward",
                "generate",
                "reset",
                "info",
                "snapshot",
                "debug_failure",
            )
        }
    )
    runtime = SimpleNamespace(_native=native)
    assert _bind(runtime).create is native.venturi_qwen_batch_create
    with pytest.raises(RuntimeError, match="rebuilt Qwen library"):
        _bind(runtime, q5_prefill_reuse=True)
    native.venturi_qwen_batch_create_q5_prefill_reuse = noop
    assert _bind(runtime, q5_prefill_reuse=True).create is noop
    assert _bind(runtime).create is native.venturi_qwen_batch_create


def _same_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _capture(batch, prompt, chunk, forced=None):
    batch.reset()
    n = (len(prompt) - 1) // chunk * chunk
    for start in range(0, n, chunk):
        batch.step([prompt[start : start + chunk]], sample_slots=[])
    scores = [batch.forward([prompt[n:]])]
    ids = []
    for step in range(3):
        token = int(scores[-1].argmax())
        ids.append(token)
        if step < 2:
            scores.append(batch.forward([[token if forced is None else forced[step]]]))
    return scores, ids, batch.snapshot(0)


@pytest.mark.cuda
def test_q5_prefill_reuse_sm70_bits_tails_and_native_multislot_guard(tmp_path):
    ops_path = os.environ.get("VENTURI_SM70_LIBRARY")
    qwen_path = os.environ.get("VENTURI_QWEN_LIBRARY")
    if not ops_path or not qwen_path or not torch.cuda.is_available():
        pytest.skip("Explicit SM70 ops/Qwen libraries and a CUDA device required")
    if torch.cuda.get_device_capability(0) != (7, 0):
        pytest.skip("SM70 only")
    config = source_config()["text_config"]
    config.update(
        hidden_size=64, head_dim=64, linear_value_head_dim=16, intermediate_size=8320
    )
    path, _, planes = write_quantized_checkpoint(
        tmp_path / "q5-prefill.safetensors", config=config, matrix_format="q5_g64_fp16"
    )
    rng = torch.Generator().manual_seed(517)
    for key, t in planes.items():
        if key.endswith((".codes", ".high")):
            t.copy_(torch.randint(256, t.shape, dtype=torch.uint8, generator=rng))
        elif key.endswith(".scales"):
            t.copy_((torch.rand(t.shape, generator=rng) * 0.02 + 0.01).half())
    write_quantized_checkpoint(
        path, config=config, matrix_format="q5_g64_fp16", updates=planes
    )
    engine = QwenRuntime(qwen_path, CudaBackend(ops_path))
    with closing(
        NativeQwenModel.from_q8(
            QwenWeights.open(path),
            engine,
            weight_mode="resident",
            dtype=torch.float16,
            device="cuda:0",
            max_tokens=32,
            reserve_bytes=0,
        )
    ) as model:
        # A direct ABI caller cannot opt into a wider batch either.
        create = _bind(engine, q5_prefill_reuse=True).create
        assert not create(model._handle, (C.c_int * 4)(*([128] * 4)), 4)
        with pytest.raises(RuntimeError, match="one slot"):
            engine._check(-1)
        with (
            closing(model.create_batch([128])) as reference,
            closing(model.create_batch([128], q5_prefill_reuse=True)) as candidate,
        ):
            for n in (16, 17, 31, 32, 65):
                prompt = [(j * 5 + 7) % candidate.vocab_size for j in range(n)]
                a, tokens, state = _capture(reference, prompt, 16)
                b, other, trial = _capture(candidate, prompt, 32, tokens)
                assert tokens == other
                for x, y in zip(a, b, strict=True):
                    _same_bits(x, y)
                for key in ("gdn", "kv"):
                    for x, y in zip(state[key], trial[key], strict=True):
                        assert (x is None) == (y is None)
                        if x is not None:
                            for u, v in zip(x, y, strict=True):
                                _same_bits(u, v)
