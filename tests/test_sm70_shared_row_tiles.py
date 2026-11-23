"""Independent packed-code oracle and opt-in native Q5 mixed dispatch."""

from contextlib import ExitStack, closing
import ctypes as C
import os

import pytest
import torch
from qwen_fixtures import source_config
from qwen_safetensors import write_quantized_checkpoint
from test_qwen_resident_embedding_io import new_model
from venturi.models import NativeQwenModel, QwenWeights
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend


@pytest.fixture(scope="module")
def backend():
    p = os.environ.get("VENTURI_TILE_OPS")
    if (
        not p
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        pytest.skip("Explicit SM70 operator library required")
    return CudaBackend(p)


@pytest.mark.cuda
@pytest.mark.parametrize("bits", (4, 5, 6))
@pytest.mark.parametrize(
    "mode,m,n,k,offset",
    [
        (1, 1, 7, 65, 1),
        (1, 2, 17, 193, 0),
        (1, 4, 64, 256, 0),
        (1, 7, 35, 257, 1),
        (2, 4, 45, 256, 1),
        (2, 9, 17, 65, 1),
        (2, 16, 128, 256, 0),
        (2, 17, 31, 129, 0),
        (2, 32, 96, 512, 0),
        (2, 33, 67, 257, 1),
        (2, 64, 128, 256, 0),
        (2, 97, 40, 144, 1),
    ],
)
def test_common_tile_matches_independent_fp64_dot(backend, bits, mode, m, n, k, offset):
    p, i = C.c_void_p, C.c_int
    fn = backend._native.venturi_row_tiles_sm70_run
    fn.argtypes, fn.restype = [p] * 5 + [i] * 6 + [p], i
    rng = torch.Generator().manual_seed(5121 + m + bits)
    padded = ((k + 127) // 128) * 128
    raw = torch.randint(1 << bits, (n, padded), dtype=torch.int32, generator=rng)
    lo = ((raw[:, ::2] & 15) | ((raw[:, 1::2] & 15) << 4)).to(torch.uint8)
    extra = bits - 4
    hi = None
    if extra:
        per_byte = 8 // extra
        hi = (
            ((raw.reshape(n, -1, per_byte) >> 4) << (torch.arange(per_byte) * extra))
            .sum(-1)
            .to(torch.uint8)
            .cuda()
        )
    scale = ((torch.rand((n, padded // 64), generator=rng) - 0.3) * 0.018).half()
    centered = torch.where(raw >= 1 << (bits - 1), raw - (1 << bits), raw)
    weight = (centered * scale.repeat_interleave(64, dim=1)).half()
    host = (torch.randn(m * k + offset, generator=rng) * 0.12).half()
    x = host.cuda()[offset:].reshape(m, k)
    dst = torch.full((m * n + 4,), -47, device="cuda", dtype=torch.float16)
    y = dst[2:-2].reshape(m, n)
    lo, sp = lo.cuda(), scale.cuda()
    args = (
        x.data_ptr(),
        lo.data_ptr(),
        hi.data_ptr() if hi is not None else None,
        sp.data_ptr(),
        y.data_ptr(),
    )
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(*args, m, n, k, padded - 1, bits, mode, stream) != 0
    assert (dst == -47).all()
    assert fn(*args, m, n, k, padded, bits, mode, stream) == 0
    want = (host[offset:].reshape(m, k).double() @ weight[:, :k].double().T).half()
    assert (dst[:2] == -47).all() and (dst[-2:] == -47).all()
    torch.testing.assert_close(y.cpu(), want, atol=1e-3, rtol=2e-3)


@pytest.mark.cuda
def test_sm70_q5_uses_shared_tile_in_actual_mixed_step(backend, tmp_path):
    q = os.environ.get("VENTURI_TILE_QWEN")
    if not q:
        pytest.skip("Explicit paired native Qwen required")
    c = source_config()["text_config"]
    c.update(
        hidden_size=128,
        head_dim=64,
        intermediate_size=4096,
        linear_key_head_dim=128,
        linear_value_head_dim=32,
        max_position_embeddings=256,
        vocab_size=4112,
    )
    artifact, _, _ = write_quantized_checkpoint(
        tmp_path / "wide.safetensors", config=c, matrix_format="q5_g64_fp16"
    )
    r = QwenRuntime(q, backend)
    with ExitStack() as stack:
        m = stack.enter_context(closing(new_model(artifact, r)))
        b = stack.enter_context(
            closing(
                m.create_batch(
                    [64, 64], resident_dataflow=True, attention_execution="descriptor"
                )
            )
        )
        b.prepare_shared_tiles(direct_outputs=True)
        with pytest.raises(RuntimeError, match="once"):
            b.prepare_dense_io()
        assert b.dense_io_info()["shared_tiles"] == 1
        b.step([[2, 3, 4]], slots=[0])
        before = b.dense_io_info()
        v = b.step([[5], [9, 2, 1, 8, 7, 3, 6]], slots=[0, 1], decode_slots=[0])
        after = b.dense_io_info()
        assert len(v) == 2 and [x["position"] for x in b.info()["requests"]] == [4, 7]
        assert after["tile_wmma_calls"] > before["tile_wmma_calls"]
        assert after["tile_grouped_calls"] > before["tile_grouped_calls"]  # head
        assert (
            after["tile_avoided_fp16_write_bytes"]
            > before["tile_avoided_fp16_write_bytes"]
        )
        assert (
            after["unpack_calls"] > before["unpack_calls"]
        )  # narrower matrices stay dense
        assert torch.isfinite(b.forward([[4], [2]], slots=[0, 1])).all()


@pytest.mark.cuda
def test_sm70_q5_modes_cannot_mix_across_batch_handles(backend, tmp_path):
    q = os.environ.get("VENTURI_TILE_QWEN")
    if not q:
        pytest.skip("Explicit paired native Qwen required")
    c = source_config()["text_config"]
    c.update(
        hidden_size=64, head_dim=64, linear_value_head_dim=16, intermediate_size=8320
    )
    artifact, _, _ = write_quantized_checkpoint(
        tmp_path / "modes.safetensors", config=c, matrix_format="q5_g64_fp16"
    )
    engine = QwenRuntime(q, backend)
    weights = QwenWeights.open(artifact)
    for exact_first in (False, True):
        with (
            closing(
                NativeQwenModel.from_q8(
                    weights,
                    engine,
                    dtype=torch.float16,
                    device="cuda:0",
                    weight_mode="resident",
                    max_tokens=32,
                    reserve_bytes=0,
                )
            ) as model,
            closing(model.create_batch([128])) as regular,
        ):
            if exact_first:
                with closing(model.create_batch([128], q5_prefill_reuse=True)):
                    with pytest.raises(RuntimeError, match="exact prefill reuse"):
                        regular.prepare_shared_tiles()
            else:
                regular.prepare_shared_tiles()
                with pytest.raises(RuntimeError, match="exact prefill reuse"):
                    model.create_batch([128], q5_prefill_reuse=True)
