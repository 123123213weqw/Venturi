"""Pair against a frozen Qwen runtime that still performed the identity gather."""

from contextlib import ExitStack, closing
import ctypes as C
import os

import pytest
import torch

from qwen_fixtures import source_config
from qwen_safetensors import write_quantized_checkpoint
from venturi.models import NativeQwenModel, QwenWeights
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend
from venturi.quantization.linear import EXECUTION_QUALIFIED


@pytest.fixture(scope="module")
def runtime_pair():
    env = (
        os.environ.get("VENTURI_IO_OPS_LIBRARY"),
        os.environ.get("VENTURI_IO_BASE_LIBRARY"),
        os.environ.get("VENTURI_IO_CANDIDATE_LIBRARY"),
    )
    if not all(env) or not torch.cuda.is_available():
        pytest.skip("Explicit frozen resident-I/O library pair required")
    ops, old, new = env
    backend = CudaBackend(ops)
    return QwenRuntime(old, backend), QwenRuntime(new, backend)


def write_fixture(path, bits):
    cfg = source_config()["text_config"]
    cfg.update(
        hidden_size=128,
        head_dim=64,
        intermediate_size=256,
        linear_key_head_dim=128,
        linear_value_head_dim=32,
        max_position_embeddings=256,
    )
    codec = f"q{bits}_g64_fp16"
    _, _, planes = write_quantized_checkpoint(path, config=cfg, matrix_format=codec)
    rng = torch.Generator().manual_seed(19919)
    for name, v in planes.items():
        if name.endswith((".codes", ".high")):
            v.copy_(torch.randint(256, v.shape, generator=rng, dtype=torch.uint8))
        if name.endswith(".scales"):
            v.copy_((torch.rand(v.shape, generator=rng) * 0.02 + 0.002).half())
    write_quantized_checkpoint(path, config=cfg, matrix_format=codec, updates=planes)


def new_model(path, runtime):
    return NativeQwenModel.from_q8(
        QwenWeights.open(path),
        runtime,
        weight_mode="resident",
        metadata_mode="sealed",
        max_tokens=8,
        dtype=torch.float16,
        device="cuda:0",
        reserve_bytes=0,
    )


def assert_exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def assert_state(a, b):
    for k in ("gdn", "kv"):
        for u, v in zip(a[k], b[k], strict=True):
            assert (u is None) == (v is None)
            if u is not None:
                for x, y in zip(u, v, strict=True):
                    assert_exact(x, y)


@pytest.mark.cuda
@pytest.mark.parametrize("bits", [4, 5, 6])
def test_direct_resident_embedding_preserves_state(bits, tmp_path, runtime_pair):
    arch = torch.cuda.get_device_capability()
    if f"q{bits}_g64_fp16" not in EXECUTION_QUALIFIED.get(arch, ()):
        pytest.skip("Do not bypass existing whole-model codec admission")
    path = tmp_path / f"q{bits}.safetensors"
    write_fixture(path, bits)
    with ExitStack() as stack:
        models = [
            stack.enter_context(closing(new_model(path, rt))) for rt in runtime_pair
        ]
        sessions = [stack.enter_context(closing(m.create_session(64))) for m in models]
        batches = [
            stack.enter_context(closing(m.create_batch([64] * 3))) for m in models
        ]
        saved = 8 * (2 * 128 + 8)
        assert (
            sessions[0].info()["request_device_bytes"]
            - sessions[1].info()["request_device_bytes"]
            == saved
        )
        a, b = (p.info() for p in batches)
        assert (
            a["shared_batch_workspace_device_bytes"]
            - b["shared_batch_workspace_device_bytes"]
            == saved
        )
        assert (
            a["all_request_device_bytes"] - b["all_request_device_bytes"] == 3 * saved
        )

        for seq in ([4, 1, 4, 12, 16], [9], [1, 1, 6]):
            assert_exact(*(p.forward(seq) for p in sessions))
        assert_state(*(p.snapshot() for p in sessions))

        grouped = [[7, 2, 7, 13], [10, 10, 1], [3]]
        assert_exact(*(p.forward(grouped, slots=[2, 1, 0]) for p in batches))
        for args in (
            {"prompts": [[9], [6, 4, 6]], "slots": [2, 0], "decode_slots": [2]},
            {"prompts": [[13], [8]], "slots": [0, 2], "decode_slots": [0, 2]},
        ):
            # Forced input IDs make this test independent of top-1 similarity.
            assert batches[0].step(**args) == batches[1].step(**args)
        for i in range(3):
            assert_state(*(p.snapshot(i) for p in batches))
        for p in [*sessions, *batches]:
            p.reset()
        assert_exact(*(p.forward([16]) for p in sessions))
        assert_exact(*(p.forward([[16]], slots=[2]) for p in batches))


@pytest.mark.cuda
@pytest.mark.parametrize("bits", [4, 5, 6])
def test_existing_indexed_dequant_writes_raw_bytes_without_identity_gather(bits):
    path = os.environ.get("VENTURI_IO_OPS_LIBRARY")
    if not path or not torch.cuda.is_available():
        pytest.skip("Explicit device operator library required")
    native = CudaBackend(path)._native
    unpack, gather = native.venturi_row_dequant_indexed, native.venturi_embedding_run
    p, i = C.c_void_p, C.c_int
    unpack.argtypes, unpack.restype = [p] * 5 + [i] * 4 + [p], i
    gather.argtypes, gather.restype = [p] * 3 + [i] * 4 + [p], i
    k, padded, n, m = 129, 256, 17, 5
    rng = torch.Generator(device="cuda").manual_seed(4906)
    lo = torch.randint(
        256, (n, padded // 2), dtype=torch.uint8, device="cuda", generator=rng
    )
    hi = torch.randint(
        256,
        (n, padded // 8 * (bits - 4)),
        dtype=torch.uint8,
        device="cuda",
        generator=rng,
    )
    sp = torch.randint(
        -32768,
        32768,
        (n, padded // 64),
        dtype=torch.int16,
        device="cuda",
        generator=rng,
    ).view(torch.float16)
    ix = torch.tensor([3, 0, 16, 3, 9], dtype=torch.int64, device="cuda")
    identity = torch.arange(m, dtype=torch.int64, device="cuda")
    tmp = torch.empty((m, k), dtype=torch.float16, device="cuda")
    old = torch.empty_like(tmp)
    new = torch.empty(m * k + 1, dtype=torch.float16, device="cuda")[1:].view(m, k)
    stream = torch.cuda.current_stream().cuda_stream
    for out in (tmp, new):
        assert (
            unpack(
                lo.data_ptr(),
                hi.data_ptr() if bits > 4 else None,
                sp.data_ptr(),
                ix.data_ptr(),
                out.data_ptr(),
                m,
                k,
                padded,
                bits,
                stream,
            )
            == 0
        )
    assert (
        gather(tmp.data_ptr(), identity.data_ptr(), old.data_ptr(), m, m, k, 0, stream)
        == 0
    )
    assert_exact(old, new)
