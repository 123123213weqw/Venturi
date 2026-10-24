"""Larger projection workspaces must preserve ragged private state and work."""

import os

import pytest
import torch

from test_completion_benchmark import load
from test_qwen_model import hybrid_fixture
from test_qwen_tensor_parallel import same_state
from test_q8_dp4a import setup
from q8_dp4a_reference import projection_reference
from venturi.models import NativeQwenModel
from venturi.ops import CudaBackend, prepare_q8_dp4a
from venturi.ops.q8_fused import prepare_q8_fused


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_PREFILL_BUDGET_LIBRARY")
    if not path:
        pytest.skip("explicit qualified projection-budget library required")
    return CudaBackend(path)


@pytest.mark.parametrize("rows", [128, 256])
@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("profile", ["fp16", "q8-residual"])
def test_large_budget_ragged_logits_cache_eos_reset_and_memory(
    tmp_path, backend, rows, tp, profile, monkeypatch
):
    weights, _ = hybrid_fixture(tmp_path)
    opts = dict(
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        reserve_bytes=0,
        projection_profile=profile,
    )
    if tp:
        opts.update(tensor_parallel_device="cuda:1", tensor_parallel_transport="host")
    models = [
        NativeQwenModel.from_q8(weights, backend, max_tokens=n, **opts)
        for n in [32, rows]
    ]
    batches = [m.create_batch([128] * 8) for m in models]
    prompts = [
        [(7 * i + j + 1) % weights.config["vocab_size"] for j in range(t)]
        for i, t in enumerate([1, 2, 7, 19, 23, 31, 33, 39])
    ]
    try:
        results = [b.forward(prompts) for b in batches]
        assert torch.equal(results[0].view(torch.uint8), results[1].view(torch.uint8))
        for i in range(8):
            same_state(batches[0].snapshot(i), batches[1].snapshot(i))
        for b in batches:
            b.reset(slots=[7, 0])
        results = [b.forward([[1, 2], [7, 3, 9]], slots=[7, 0]) for b in batches]
        assert torch.equal(results[0].view(torch.uint8), results[1].view(torch.uint8))
        for i in range(8):
            same_state(batches[0].snapshot(i), batches[1].snapshot(i))
        for b in batches:
            b.reset()
        maximum = [1, 2, 3, 4, 5, 6, 7, 8]
        expected = batches[0].generate(
            prompts,
            max_new_tokens=maximum,
            eos_token_ids=[[], [], [1], [], [2], [], [], []],
        )

        def forbidden(*a, **kw):
            raise AssertionError(
                "Native larger-budget generation used Torch math/allocation"
            )

        with monkeypatch.context() as p:
            for name in ["empty", "zeros", "matmul", "mm", "bmm", "argmax"]:
                p.setattr(torch, name, forbidden)
            actual = batches[1].generate(
                prompts,
                max_new_tokens=maximum,
                eos_token_ids=[[], [], [1], [], [2], [], [], []],
            )
        assert actual == expected
        for i in range(8):
            same_state(batches[0].snapshot(i), batches[1].snapshot(i))
        info = [b.info() for b in batches]
        assert info[1]["prefill_group_calls"] < info[0]["prefill_group_calls"]
        assert (
            info[1]["max_group_rows"] <= rows
            and info[1]["input_rows"] == info[0]["input_rows"]
        )
        for i in range(8):
            assert (
                info[1]["requests"][i]["position"] == info[0]["requests"][i]["position"]
            )
        # A bigger workspace is not free memory or a smaller per-request context.
        assert (
            info[1]["requests"][0]["request_device_bytes"]
            > info[0]["requests"][0]["request_device_bytes"]
        )
    finally:
        for b in batches:
            b.close()
        for m in models:
            m.close()


@pytest.mark.parametrize("tokens", [56, 127, 128, 184, 255, 256])
@pytest.mark.parametrize("k", [67, 5120])
@pytest.mark.parametrize("ns", [(17, 33, 7), (1025, 17)])
def test_large_fused_token_tails_and_independent_fp64(backend, tokens, k, ns):
    data = [setup(backend, n, k, tokens) for n in ns]
    x = data[0][2].cuda()
    res = [d[3] for d in data]
    p = prepare_q8_fused(
        res,
        tokens=tokens,
        residual_compensation=True,
        output_strides=[n + 5 for n in ns],
    )
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    outs = [
        torch.full((tokens, s), 47, dtype=torch.float16, device="cuda:0")
        for s in p.strides
    ]
    p.run(x, outputs=outs, workspace=work)
    packed = work.clone()
    for d, n, out in zip(data, ns, outs):
        old = prepare_q8_dp4a(d[3], tokens=tokens, residual_compensation=True)
        expected = torch.empty(tokens, n, dtype=torch.float16, device="cuda:0")
        old.run(x, out=expected, workspace=work)
        assert torch.equal(
            out[:, :n].contiguous().view(torch.uint8), expected.view(torch.uint8)
        )
        assert (out[:, n:] == 47).all() and torch.equal(work, packed)
        sample = list(dict.fromkeys([0, min(13, n - 1), n - 1]))
        ref = projection_reference(data[0][2], d[0][sample], d[1][sample], 2)
        assert (expected[:, sample].cpu().double() - ref).norm() / (
            ref.norm() + 1e-12
        ) < 0.001


@pytest.mark.parametrize(
    "tool",
    ["serve_native_benchmark", "profile_qwen_batch", "qualify_qwen_projection_variant"],
)
def test_projection_budget_cli_rejects_invalid_before_runtime(
    tool, monkeypatch, tmp_path
):
    import sys

    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = load(tool)
    monkeypatch.setattr(sys, "argv", [tool, "--projection-rows", "0"])
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code == 2
