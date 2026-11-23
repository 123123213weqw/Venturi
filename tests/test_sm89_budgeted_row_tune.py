"""Explicit SM89 FP16 row tiles: masks, independent FP64 reference and native table."""

from contextlib import ExitStack, closing
import ctypes as C
import os

import pytest
import torch
from qwen_fixtures import source_config
from qwen_safetensors import write_quantized_checkpoint
from test_qwen_resident_embedding_io import new_model
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend


@pytest.fixture(scope="module")
def ada():
    ops = os.environ.get("VENTURI_ADA_TILE_OPS")
    qwen = os.environ.get("VENTURI_ADA_TILE_QWEN")
    if (
        not ops
        or not qwen
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 9)
    ):
        pytest.skip("Explicit paired SM89 libraries required")
    return CudaBackend(ops), qwen


@pytest.mark.cuda
@pytest.mark.parametrize(
    "mode,m,n,k",
    [
        (1, 1, 7, 65),
        (1, 4, 35, 257),
        (1, 8, 64, 256),
        (2, 4, 45, 257),
        (2, 16, 128, 256),
        (2, 17, 31, 129),
        (2, 32, 96, 512),
        (2, 33, 67, 257),
        (2, 97, 40, 144),
    ],
)
def test_sm89_row_tiles_against_fp64(ada, mode, m, n, k):
    engine, _ = ada
    padded = ((k + 127) // 128) * 128
    rng = torch.Generator().manual_seed(2513 + m)
    raw = torch.randint(32, (n, padded), generator=rng, dtype=torch.int32)
    low = ((raw[:, ::2] & 15) | ((raw[:, 1::2] & 15) << 4)).byte().cuda()
    high = ((raw.reshape(n, -1, 8) >> 4) << torch.arange(8)).sum(-1).byte().cuda()
    scales = (0.015 * torch.rand((n, padded // 64), generator=rng)).half()
    weights = (
        torch.where(raw < 16, raw, raw - 32) * scales.repeat_interleave(64, dim=1)
    ).half()[:, :k]
    x0 = (0.2 * torch.randn((m, k), generator=rng)).half()
    out = torch.full((m * n + 4,), -12, dtype=torch.float16, device="cuda")
    x, s = x0.cuda(), scales.cuda()
    fn = engine._native.venturi_row_tiles_sm89_run
    fn.argtypes, fn.restype = [C.c_void_p] * 5 + [C.c_int] * 6 + [C.c_void_p], C.c_int
    st = torch.cuda.current_stream().cuda_stream
    assert (
        fn(
            x.data_ptr(),
            low.data_ptr(),
            high.data_ptr(),
            s.data_ptr(),
            out[2:].data_ptr(),
            m,
            n,
            k,
            padded,
            5,
            mode,
            st,
        )
        == 0
    )
    torch.testing.assert_close(
        out[2:-2].reshape(m, n).cpu(),
        (x0.double() @ weights.double().T).half(),
        atol=1e-3,
        rtol=2e-3,
    )
    assert (out[:2] == -12).all() and (out[-2:] == -12).all()


@pytest.mark.cuda
def test_sm89_explicit_table_applies_and_keeps_dense_for_unlisted_shapes(ada, tmp_path):
    eng, qwen = ada
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
    with ExitStack() as stack:
        m = stack.enter_context(closing(new_model(artifact, QwenRuntime(qwen, eng))))
        b = stack.enter_context(
            closing(
                m.create_batch(
                    [64, 64], resident_dataflow=True, attention_execution="descriptor"
                )
            )
        )
        with pytest.raises(RuntimeError, match="duplicate"):
            b.prepare_ada_tiles([(1, 4112, 128, 1)] * 2)
        b.prepare_ada_tiles([(1, 4112, 128, 1), (4, 4096, 128, 2)], direct_outputs=True)
        b.step([[2]], slots=[0])
        b.step([[7], [2, 3, 4]], slots=[0, 1], sample_slots=[0], decode_slots=[0])
        z = b.dense_io_info()
        assert (
            z["tile_grouped_calls"] > 0
            and z["tile_wmma_calls"] > 0
            and z["unpack_calls"] > 0
        )
        assert [r["position"] for r in b.info()["requests"]] == [2, 3]
        plain = stack.enter_context(
            closing(
                m.create_batch(
                    [64, 64], resident_dataflow=True, attention_execution="descriptor"
                )
            )
        )
        plain.step([[3]], slots=[0])
        untouched = plain.dense_io_info()
        assert untouched["tile_wmma_calls"] == z["tile_wmma_calls"]
        assert untouched["tile_grouped_calls"] == z["tile_grouped_calls"]
        assert untouched["unpack_calls"] > z["unpack_calls"]
        with pytest.raises(RuntimeError, match="before this batch projects"):
            plain.prepare_ada_tiles([(1, 4112, 128, 1)], direct_outputs=True)
        assert torch.isfinite(b.forward([[5], [6]], slots=[0, 1])).all()
