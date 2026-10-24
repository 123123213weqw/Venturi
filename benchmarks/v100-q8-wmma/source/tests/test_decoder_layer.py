"""Decoder composition, ownership, KV state transitions and workspace liveness."""

from dataclasses import replace

import pytest
import torch

from decoder_reference import decoder_reference
from test_gated_mlp import make_weights
from venturi.models import AttentionDimensions, DecoderLayerWeights, DecoderWorkspace
from venturi.ops import DenseWeight, Q4RowSplitWeight, TorchBackend


def layer_weights(
    h=16, intermediate=32, dims=None, dtype=torch.float64, q4=False, seed=8
):
    dims = dims or AttentionDimensions(2, 1, 64)
    gen = torch.Generator().manual_seed(seed)

    def weight(n, k):
        if q4:
            codes = torch.randint(-8, 8, (n, k), dtype=torch.int8, generator=gen)
            scales = torch.full((n, (k + 63) // 64), 0.15 / k**0.5, dtype=torch.float16)
            return Q4RowSplitWeight.from_codes(codes, scales)
        return DenseWeight(
            (torch.randn(n, k, generator=gen, dtype=dtype) / k**0.5).contiguous()
        )

    return DecoderLayerWeights(
        torch.linspace(0.8, 1.2, h, dtype=dtype),
        weight(dims.query_width, h),
        weight(dims.kv_width, h),
        weight(dims.kv_width, h),
        weight(h, dims.query_width),
        make_weights(h, intermediate, dtype=dtype, q4=q4),
    )


def rope_tables(tokens, dims, device="cpu"):
    # Deliberately nontrivial, explicitly selected absolute positions.
    position = torch.arange(tokens, dtype=torch.float64)[:, None]
    freq = 10000 ** (
        -torch.arange(dims.rotary_dim // 2, dtype=torch.float64)
        / (dims.rotary_dim // 2)
    )
    angle = position * freq
    return angle.cos().float().to(device), angle.sin().float().to(device)


@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
@pytest.mark.parametrize("q4", [False, True])
def test_fp64_prefill_chunk_decode_against_independent_reference(layout, q4):
    dims = AttentionDimensions(4, 2, 64, rotary_dim=32, rope_layout=layout)
    weights = layer_weights(dims=dims, q4=q4)
    resident = weights.upload(
        TorchBackend(), dimensions=dims, dtype=torch.float64, device="cpu"
    )
    cos, sin = rope_tables(9, dims)
    x = torch.randn(
        9, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(27)
    )
    expected, expected_cache = decoder_reference(weights, dims, x, cos, sin)
    for chunks in ([9], [3, 2, 4], [1] * 9):
        cache = resident.empty_cache(capacity=12)
        # The exact same arena also serves all smaller token shapes.
        arena = resident.prepare(tokens=9).empty_workspace()
        outputs, position = [], 0
        for count in chunks:
            plan = resident.prepare(tokens=count)
            step = plan.at(position=position)
            out = torch.empty(count, 16, dtype=x.dtype)
            step.run(
                x[position : position + count],
                cos[position : position + count],
                sin[position : position + count],
                out=out,
                workspace=arena,
                cache=cache,
            )
            outputs.append(out)
            position += count
            assert cache.length == position
            assert not cache.failed
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(
            cache.keys, expected_cache[0], rtol=2e-12, atol=2e-12
        )
        torch.testing.assert_close(
            cache.values, expected_cache[1], rtol=2e-12, atol=2e-12
        )
        assert cache.nbytes == 2 * 12 * dims.kv_width * 8


def test_shared_weights_arena_liveness_and_position_only_binding(monkeypatch):
    weights = layer_weights(dtype=torch.bfloat16)
    resident = weights.upload(
        TorchBackend(), dimensions=AttentionDimensions(2, 1, 64), device="cpu"
    )
    before = tuple(t.data_ptr() for t in resident.storage_tensors)

    def forbidden(*args, **kwargs):
        raise AssertionError("shape/position binding copied or allocated a tensor")

    with monkeypatch.context() as m:
        for name in ("to", "clone", "contiguous"):
            m.setattr(torch.Tensor, name, forbidden)
        m.setattr(torch, "empty", forbidden)
        plans = [resident.prepare(tokens=t) for t in (1, 3, 7)]
        steps = [plans[0].at(position=p) for p in (0, 3, 7)]
    assert all(tuple(t.data_ptr() for t in p.storage_tensors) == before for p in plans)
    assert all(s.layer is plans[0] for s in steps)
    arena = plans[-1].empty_workspace()
    hidden, residual, q, attended, mlp = arena._views(plans[0])
    assert q.data_ptr() == mlp.gate.data_ptr()
    assert attended.data_ptr() == mlp.up.data_ptr()
    assert hidden.data_ptr() == mlp.hidden.data_ptr()
    assert residual.data_ptr() != hidden.data_ptr()
    assert all(
        (t.data_ptr() - hidden.data_ptr()) % 256 == 0
        for t in (hidden, residual, q, attended)
    )
    # Arena is independent of the visible KV prefix length.
    assert plans[0].at(position=200).layer is plans[0]


def test_two_layers_reuse_one_arena_without_reusing_kv_or_weights():
    dims = AttentionDimensions(2, 1, 64)
    weights = [layer_weights(seed=s) for s in (8, 9)]
    residents = [
        w.upload(TorchBackend(), dimensions=dims, dtype=torch.float64, device="cpu")
        for w in weights
    ]
    plans = [r.prepare(tokens=5) for r in residents]
    arena = plans[0].empty_workspace()
    caches = [r.empty_cache(capacity=8) for r in residents]
    x = torch.randn(5, 16, dtype=torch.float64)
    cos, sin = rope_tables(5, dims)
    expected = x
    current = x
    for weight, plan, cache in zip(weights, plans, caches):
        expected, _ = decoder_reference(weight, dims, expected, cos, sin)
        out = torch.empty_like(x)
        plan.at(position=0).run(
            current, cos, sin, out=out, workspace=arena, cache=cache
        )
        current = out
    torch.testing.assert_close(current, expected, rtol=2e-12, atol=2e-12)
    assert caches[0]._keys.data_ptr() != caches[1]._keys.data_ptr()


def test_rejected_inputs_do_not_mutate_cache_or_output():
    dims = AttentionDimensions(2, 1, 64)
    resident = layer_weights().upload(
        TorchBackend(), dimensions=dims, dtype=torch.float64, device="cpu"
    )
    plan = resident.prepare(tokens=3)
    step = plan.at(position=0)
    cache, arena = resident.empty_cache(capacity=4), plan.empty_workspace()
    cache._keys.fill_(47)
    cache._values.fill_(47)
    x, out = (
        torch.ones(3, 16, dtype=torch.float64),
        torch.full((3, 16), 47.0, dtype=torch.float64),
    )
    cos, sin = rope_tables(3, dims)
    cases = [
        dict(out=x),
        dict(cos=cos.double()),
        dict(workspace=replace(arena, max_tokens=1)),
        dict(workspace=replace(arena, activation=cache._keys.flatten())),
        dict(cache=resident.empty_cache(capacity=2)),
        dict(x=x.clone().requires_grad_()),
    ]
    for changes in cases:
        kwargs = dict(x=x, cos=cos, sin=sin, out=out, workspace=arena, cache=cache)
        kwargs.update(changes)
        with pytest.raises((ValueError, TypeError)):
            step.run(**kwargs)
        assert cache.length == 0 and not cache.failed
        assert bool((cache._keys == 47).all()) and bool((cache._values == 47).all())
        assert bool((out == 47).all())
    with pytest.raises(ValueError, match="visible length"):
        plan.at(position=1).run(x, cos, sin, out=out, workspace=arena, cache=cache)
    other = layer_weights().upload(
        TorchBackend(), dimensions=dims, dtype=torch.float64, device="cpu"
    )
    with pytest.raises(ValueError, match="belong"):
        step.run(
            x, cos, sin, out=out, workspace=arena, cache=other.empty_cache(capacity=4)
        )


def test_failure_reset_truncate_capacity_and_source_isolation(monkeypatch):
    dims = AttentionDimensions(2, 1, 64)
    weights = layer_weights()
    x = torch.randn(4, 16, dtype=torch.float64)
    cos, sin = rope_tables(4, dims)
    expected, _ = decoder_reference(weights, dims, x, cos, sin)
    resident = weights.upload(
        TorchBackend(), dimensions=dims, dtype=torch.float64, device="cpu"
    )
    plan, cache = resident.prepare(tokens=4), resident.empty_cache(capacity=4)
    arena, out = plan.empty_workspace(), torch.empty_like(x)
    # Source descriptors may be discarded/mutated after an owned upload.
    weights.attention_norm.zero_()
    for weight in (weights.query, weights.key, weights.value, weights.output):
        weight.data.zero_()
    step = plan.at(position=0)

    original_run = type(plan.output).run

    def broken(operation, *args, **kwargs):
        if operation is plan.output:
            raise RuntimeError("injected submission failure after KV projection")
        return original_run(operation, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(type(plan.output), "run", broken)
        with pytest.raises(RuntimeError, match="injected"):
            step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
    assert cache.failed and cache.length == 0
    with pytest.raises(RuntimeError, match="reset"):
        step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
    cache.reset()
    step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
    torch.testing.assert_close(out, expected, rtol=2e-12, atol=2e-12)
    assert cache.length == cache.capacity
    with pytest.raises(ValueError, match="capacity"):
        resident.prepare(tokens=1).at(position=4).run(
            x[:1], cos[:1], sin[:1], out=out[:1], workspace=arena, cache=cache
        )
    cache.truncate(2)
    assert cache.length == 2
    for bad in (-1, 3, True):
        with pytest.raises(ValueError):
            cache.truncate(bad)
    cache.reset()
    assert cache.length == 0
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        step.run(x, cos, sin, out=out, workspace=arena, cache=cache)


@pytest.mark.parametrize(
    "config",
    [
        dict(query_heads=3, kv_heads=2, head_dim=64),
        dict(query_heads=2, kv_heads=1, head_dim=65),
        dict(query_heads=2, kv_heads=1, head_dim=64, rotary_dim=3),
    ],
)
def test_invalid_dimensions(config):
    with pytest.raises(ValueError):
        AttentionDimensions(**config)


def test_descriptor_and_precision_validation():
    weights = layer_weights()
    dims = AttentionDimensions(2, 1, 64)
    with pytest.raises(ValueError, match="BF16"):
        weights.upload(
            TorchBackend(), dimensions=dims, dtype=torch.float32, device="cpu"
        )
    with pytest.raises(ValueError, match="Projection widths"):
        weights.upload(
            TorchBackend(),
            dimensions=AttentionDimensions(4, 1, 64),
            dtype=torch.float64,
            device="cpu",
        )
    with pytest.raises(ValueError, match="Q/K/V"):
        replace(weights, key=DenseWeight(torch.ones(64, 17)))
    assert isinstance(
        resident_workspace := weights.upload(
            TorchBackend(), dimensions=dims, dtype=torch.float64, device="cpu"
        )
        .prepare(tokens=1)
        .empty_workspace(),
        DecoderWorkspace,
    )
    assert resident_workspace.nbytes > 0


def test_host_submissions_to_one_arena_are_serialized_across_requests():
    from concurrent.futures import ThreadPoolExecutor

    dims = AttentionDimensions(2, 1, 64)
    weights = layer_weights()
    resident = weights.upload(
        TorchBackend(), dimensions=dims, dtype=torch.float64, device="cpu"
    )
    plan = resident.prepare(tokens=3)
    arena = plan.empty_workspace()
    cos, sin = rope_tables(3, dims)
    inputs = [torch.randn(3, 16, dtype=torch.float64) for _ in range(12)]
    expected = [decoder_reference(weights, dims, x, cos, sin)[0] for x in inputs]

    def request(x):
        cache = resident.empty_cache(capacity=3)
        out = torch.empty_like(x)
        plan.at(position=0).run(x, cos, sin, out=out, workspace=arena, cache=cache)
        return out

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(request, inputs))
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual, wanted, rtol=2e-12, atol=2e-12)
