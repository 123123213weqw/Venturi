"""Canonical rounding with a split head, direct outputs and bounded tile shadows."""

from contextlib import ExitStack, closing
import os

import pytest
import torch
from qwen_fixtures import source_config
from qwen_safetensors import write_quantized_checkpoint
from test_qwen_resident_embedding_io import assert_exact, assert_state, new_model
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend


@pytest.mark.cuda
def test_dense_staging_preserves_split_head_rounding_and_kv(tmp_path):
    op, runtime = (
        os.environ.get(k)
        for k in ("VENTURI_DENSE_OPS_LIBRARY", "VENTURI_DENSE_QWEN_LIBRARY")
    )
    if (
        not op
        or not runtime
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        pytest.skip("Explicit SM70 native pair required")
    cfg = source_config()["text_config"]
    cfg.update(
        hidden_size=128,
        head_dim=64,
        intermediate_size=256,
        linear_key_head_dim=128,
        linear_value_head_dim=32,
        max_position_embeddings=256,
        vocab_size=8201,
    )
    artifact = tmp_path / "split-head.safetensors"
    _, _, planes = write_quantized_checkpoint(
        artifact, config=cfg, matrix_format="q5_g64_fp16"
    )
    rng = torch.Generator().manual_seed(8494)
    for k, v in planes.items():
        if k.endswith((".codes", ".high")):
            v.copy_(torch.randint(256, v.shape, dtype=torch.uint8, generator=rng))
        elif k.endswith(".scales"):
            v.copy_((torch.rand(v.shape, generator=rng) * 0.018 + 0.002).half())
    write_quantized_checkpoint(
        artifact, config=cfg, matrix_format="q5_g64_fp16", updates=planes
    )

    rt = QwenRuntime(runtime, CudaBackend(op))
    with ExitStack() as stack:
        models = [
            stack.enter_context(closing(new_model(artifact, rt))) for _ in range(3)
        ]
        base, direct, cache = [
            stack.enter_context(closing(m.create_batch([64, 64]))) for m in models
        ]
        size = base.dense_io_info()
        assert size["workspace_bytes"] < size["prior_workspace_bytes"]
        assert size["weight_scratch_bytes"] == 8192 * 128 * 2
        direct.prepare_dense_io(direct_outputs=True)
        cache.prepare_dense_io(direct_outputs=True, shadow_budget_bytes=125_000)
        plan = cache.dense_io_info()
        assert 0 < plan["shadow_bytes"] <= 125_000 and plan["shadow_tiles"] > 0
        assert (
            plan["workspace_bytes"] + plan["shadow_bytes"]
            < plan["prior_workspace_bytes"]
        )
        with pytest.raises(RuntimeError, match="once"):
            cache.prepare_dense_io(direct_outputs=True)

        batches = (base, direct, cache)
        expected = [b.forward([[8200, 8192, 3], [8193, 4, 8200]]) for b in batches]
        for y in expected[1:]:
            assert_exact(y, expected[0])
        with pytest.raises(RuntimeError, match="before the first"):
            base.prepare_dense_io(direct_outputs=True)
        d = direct.dense_io_info()
        assert 0 < d["direct_gemms"] < d["gemms"]  # split multi-row head still stages
        for n in range(3):
            seq = [[8200 - n], [8199 - n]]
            ans = [b.step(seq, slots=[1, 0], decode_slots=[1, 0]) for b in batches]
            assert all(x == ans[0] for x in ans)
        before = [b.dense_io_info() for b in batches]
        one = [b.step([[8]], slots=[1], decode_slots=[1]) for b in batches]
        assert all(x == one[0] for x in one)
        after = [b.dense_io_info() for b in batches]
        assert (
            after[1]["gemms"] - before[1]["gemms"]
            == after[1]["direct_gemms"] - before[1]["direct_gemms"]
        )
        assert after[1]["output_copy_bytes"] == before[1]["output_copy_bytes"]
        assert 0 < after[2]["unpack_bytes"] < after[0]["unpack_bytes"]
        assert (
            after[2]["unpack_bytes"] + after[2]["shadow_read_bytes"]
            == after[0]["unpack_bytes"]
        )
        for i in range(2):
            for b in (direct, cache):
                assert_state(base.snapshot(i), b.snapshot(i))
