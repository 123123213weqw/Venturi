"""A semantic head split is not a flat half-row QKV slice or one long sequence."""

from contextlib import ExitStack
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from qwen_gdn_reference import gdn_layer_oracle
from test_qwen_model import hybrid_fixture


@pytest.fixture
def pilot(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    import gdn_head_pilot

    return gdn_head_pilot


def test_actual_semantic_intervals_and_inverse_history_join(pilot):
    ranges = [pilot.semantic_rows(16, 48, 128, 128, rank) for rank in (0, 1)]
    assert ranges[0] == tuple(
        list(range(0, 1024)) + list(range(2048, 3072)) + list(range(4096, 7168))
    )
    assert ranges[1] == tuple(
        list(range(1024, 2048)) + list(range(3072, 4096)) + list(range(7168, 10240))
    )
    source = torch.arange(2 * 3 * 10240).view(2, 3, 10240)
    local = [source[..., list(ids)] for ids in ranges]
    assert torch.equal(pilot.join_channels(local, 16, 48, 128, 128), source)
    assert not torch.equal(torch.cat(local, -1), source)


@pytest.mark.parametrize(
    "shape", [(1, 3, 128, 128), (3, 9, 128, 128), (2, 5, 128, 128)]
)
def test_invalid_semantic_groups_rejected(pilot, shape):
    with pytest.raises(ValueError, match="Semantic"):
        pilot.semantic_rows(*shape, 0)


def test_encoded_rows_and_scale_planes_preserved(tmp_path, pilot):
    w, _ = hybrid_fixture(tmp_path)
    p = pilot.Parts.from_weights(w)
    f = p.frontend
    for rank in (0, 1):
        ids = pilot.semantic_rows(
            f.qk_heads, f.value_heads, f.key_dim, f.value_dim, rank
        )
        local = pilot.take_rows(p.qkv, ids)
        assert torch.equal(local.codes, p.qkv.codes[list(ids)])
        assert torch.equal(local.scales, p.qkv.scales[list(ids)])


def test_reference_gdn_stop_preserves_existing_complete_layer_default(tmp_path):
    w, raw = hybrid_fixture(tmp_path)
    cfg = w.config
    x = torch.randn(
        3, cfg["hidden_size"], generator=torch.Generator().manual_seed(78)
    ).half()
    c = (
        2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    )
    hist = torch.zeros(cfg["linear_conv_kernel_dim"] - 1, c, dtype=torch.float16)
    state = torch.zeros(
        cfg["linear_num_value_heads"],
        cfg["linear_value_head_dim"],
        cfg["linear_key_head_dim"],
        dtype=torch.float32,
    )
    whole, h, s, trace = gdn_layer_oracle(raw, cfg, x, hist, state)
    block, h2, s2, t = gdn_layer_oracle(raw, cfg, x, hist, state, only_gdn=True)
    assert (
        torch.equal(block, trace["projected"])
        and torch.equal(h, h2)
        and torch.equal(s, s2)
    )
    assert "mlp_down" in trace and "mlp_down" not in t and whole.shape == block.shape


@pytest.fixture
def libraries():
    paths = [
        os.environ.get(k) for k in ("VENTURI_GDN_HEAD_BLOCKS", "VENTURI_GDN_HEAD_HOST")
    ]
    if not all(paths):
        pytest.skip("Explicit qualified blocks and research native-host SO required")
    return paths


def inputs(parts, sequences, steps, seed=411):
    g = torch.Generator().manual_seed(seed)
    f = parts.frontend
    return (
        torch.randn(sequences * steps, parts.qkv.in_features, generator=g).half(),
        (
            torch.randn(sequences, f.conv.shape[0] - 1, f.conv.shape[1], generator=g)
            * 0.1
        ).half(),
        torch.randn(sequences, f.value_heads, f.value_dim, f.key_dim, generator=g)
        * 0.001,
    )


def error(actual, expected):
    return float(
        (actual.double() - expected.double()).norm()
        / expected.double().norm().clamp(min=1e-12)
    )


def validate(row, head):
    a, b = row.snapshot(), head.snapshot()
    for name in a:
        if name not in ("output", "projected"):
            assert torch.equal(a[name].view(torch.uint8), b[name].view(torch.uint8)), (
                name
            )
    assert error(a["output"], b["output"]) < 5e-4
    assert error(a["projected"], b["projected"]) < 5e-4
    return a, b


@pytest.mark.parametrize("sequences,steps", [(1, 1), (1, 3), (8, 1)])
def test_private_states_prefixes_and_output_against_independent_reference(
    tmp_path, pilot, libraries, sequences, steps
):
    w, raw = hybrid_fixture(tmp_path)
    parts = pilot.Parts.from_weights(w)
    x, hist, state = inputs(parts, sequences, steps)
    plans = [
        pilot.Pilot(
            parts,
            *libraries,
            mode=mode,
            sequences=sequences,
            steps=steps,
            inputs=x,
            histories=hist,
            states=state,
        )
        for mode in ("row", "head")
    ]
    try:
        for p in plans:
            p.run()
        a, b = validate(*plans)
        for s in range(sequences):
            ideal, _, _, _ = gdn_layer_oracle(
                raw,
                w.config,
                x[s * steps : (s + 1) * steps],
                hist[s],
                state[s],
                only_gdn=True,
            )
            assert error(b["projected"][s * steps : (s + 1) * steps], ideal) < 0.02
            assert (
                error(
                    b["output"][s * steps : (s + 1) * steps],
                    (x[s * steps : (s + 1) * steps].double() + ideal.double()).half(),
                )
                < 0.02
            )
        # Long repeated ownership cycles cannot reuse stale pinned/peer scratch.
        for p in plans:
            p.run(11)
        validate(*plans)
        for p in plans:
            p.reset()
            p.run()
        a2, b2 = validate(*plans)
        assert all(torch.equal(a[k], a2[k]) for k in a)
        if sequences == 8:
            before = plans[1].snapshot()
            plans[1].reset([7, 0])
            after = plans[1].snapshot()
            assert torch.equal(before["state"][1:7], after["state"][1:7])
            assert torch.equal(before["history"][1:7], after["history"][1:7])
            plans[0].reset([7, 0])
            for p in plans:
                p.run()
            validate(*plans)

        def forbidden(*args, **kwargs):
            raise AssertionError("Python numerical operation in native pilot loop")

        with ExitStack() as stack:
            for name in ("empty", "matmul", "mm", "zeros", "tensor", "cat"):
                stack.enter_context(patch.object(torch, name, forbidden))
            for p in plans:
                p.run(2)
        validate(*plans)
        plans[1].close()
        with pytest.raises(RuntimeError, match="closed"):
            plans[1].run()
    finally:
        for p in plans:
            p.close()
