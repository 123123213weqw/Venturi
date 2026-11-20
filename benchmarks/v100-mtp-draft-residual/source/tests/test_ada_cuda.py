"""Opt-in native tests. Never compile or select a GPU implicitly."""

import os

import pytest
import torch

from venturi.ops import AdaBackend, Q4RowSplitWeight


pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def backend():
    library = os.environ.get("VENTURI_ADA_LIBRARY")
    if not library:
        pytest.skip("set VENTURI_ADA_LIBRARY to an explicitly built native library")
    if not torch.cuda.is_available():
        pytest.fail("VENTURI_ADA_LIBRARY is set but CUDA is unavailable")
    return AdaBackend(library)


def make_weight(n, k):
    generator = torch.Generator().manual_seed(71 + n + k)
    codes = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
    scales = (torch.rand((n, (k + 63) // 64), generator=generator) * 0.2 + 0.001).half()
    return Q4RowSplitWeight.from_codes(codes, scales)


def oracle(weight, x):
    # Deliberately decode independently of Q4RowSplitWeight.dequantize_reference.
    # FP64 reference retains exact stored scales and exact activation values.
    packed = weight.codes.to(torch.int16)
    low = packed.remainder(16)
    high = packed.div(16, rounding_mode="floor")
    values = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.int16)
    values[:, ::2], values[:, 1::2] = low, high
    values[values >= 8] -= 16
    values = values.to(torch.float64).view(weight.out_features, -1, 64)
    values *= weight.scales.double().unsqueeze(-1)
    dense = values.flatten(1)[:, : weight.in_features]
    return x.cpu().double() @ dense.T


def assert_numerics(actual, expected, dtype):
    # Bound FP32 accumulation error plus one output-dtype rounding interval.
    rounded = expected.to(dtype).double()
    error = (actual.cpu().double() - rounded).abs()
    epsilon = torch.finfo(dtype).eps
    bound = 2 * epsilon * expected.abs() + 2e-4
    assert bool((error <= bound).all()), f"max error={float(error.max())}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "n,k,t",
    [
        (1, 1, 1),
        (37, 65, 3),
        (41, 127, 9),
        (33, 129, 17),
        (137, 4096, 33),
        (80, 5120, 1),
    ],
)
def test_general_shapes(backend, dtype, n, k, t):
    weight = make_weight(n, k)
    plan = backend.prepare_linear(weight, tokens=t, dtype=dtype)
    x = torch.randn((t, k), dtype=dtype, device="cuda")
    out = plan.empty_output()
    assert plan.run(x, out=out) is out
    assert plan.route == "generic-simt"
    assert_numerics(out, oracle(weight, x), dtype)


@pytest.mark.parametrize(
    "k,t",
    [
        (5120, 2),
        (5120, 8),
        (5120, 9),
        (5120, 16),
        (6144, 8),
        (6144, 16),
        (25600, 8),
        (25600, 16),
    ],
)
def test_retained_mma_schedule(backend, k, t):
    weight = make_weight(80, k)
    plan = backend.prepare_linear(weight, tokens=t, route="tuned")
    x = torch.randn((t, k), dtype=torch.bfloat16, device="cuda")
    out = plan.empty_output()
    plan.run(x, out=out)
    assert plan.route == "ada-mma-small-t"
    assert_numerics(out, oracle(weight, x), torch.bfloat16)


@pytest.mark.parametrize(
    "n,k,t",
    [(32, 512, 1), (64, 1024, 3), (96, 3584, 9), (128, 4096, 17), (64, 8192, 33)],
)
def test_general_mma_dimensions(backend, n, k, t):
    weight = make_weight(n, k)
    plan = backend.prepare_linear(weight, tokens=t, route="mma")
    x = torch.randn((t, k), dtype=torch.bfloat16, device="cuda")
    out = plan.empty_output()
    plan.run(x, out=out)
    assert plan.route == "ada-mma-general"
    assert_numerics(out, oracle(weight, x), torch.bfloat16)


@pytest.mark.parametrize("n,k,t", [(4096, 4096, 1), (4096, 4096, 17), (5120, 25600, 8)])
def test_full_size_operator(backend, n, k, t):
    weight = make_weight(n, k)
    plan = backend.prepare_linear(weight, tokens=t)
    x = torch.randn((t, k), dtype=torch.bfloat16, device="cuda")
    out = plan.empty_output()
    plan.run(x, out=out)
    # Check every row in bounded CPU tiles, not only selected output entries.
    for start in range(0, n, 128):
        part = Q4RowSplitWeight(
            weight.codes[start : start + 128], weight.scales[start : start + 128], k
        )
        assert_numerics(out[:, start : start + 128], oracle(part, x), torch.bfloat16)


@pytest.mark.parametrize(
    "n,k,t,dtype",
    [
        (37, 65, 3, torch.float16),
        (80, 5120, 8, torch.bfloat16),
        (128, 4096, 17, torch.bfloat16),
    ],
)
def test_stream_graph_and_no_allocations(backend, n, k, t, dtype, monkeypatch):
    weight = make_weight(n, k)
    plan = backend.prepare_linear(weight, tokens=t, dtype=dtype)
    x = torch.randn((t, k), dtype=dtype, device="cuda")
    out = plan.empty_output()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run(x, out=out)
    stream.synchronize()
    assert_numerics(out, oracle(weight, x), dtype)
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run(x, out=out)
        x.fill_(0.25)
        graph.replay()
    stream.synchronize()
    assert_numerics(out, oracle(weight, x), dtype)

    # Native launch must not call an allocating tensor helper.
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation helper called by run")

    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        for _ in range(20):
            plan.run(x, out=out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.max_memory_allocated() == allocated


def test_owned_preparation_and_operand_errors(backend):
    weight = make_weight(128, 128)
    plan = backend.prepare_linear(weight, tokens=2)
    expected_weight = make_weight(128, 128)
    weight.codes.zero_()
    weight.scales.zero_()
    x = torch.randn((2, 128), dtype=torch.bfloat16, device="cuda")
    out = plan.empty_output()
    plan.run(x, out=out)
    assert_numerics(out, oracle(expected_weight, x), torch.bfloat16)
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=x)
    for bad in (
        x.float(),
        x.cpu(),
        x[:, :64],
        torch.empty(128, 2, dtype=x.dtype, device=x.device).T,
    ):
        with pytest.raises(ValueError):
            plan.run(bad, out=out)
    with pytest.raises(ValueError, match="inference-only"):
        plan.run(x.requires_grad_(), out=out)
    with pytest.raises(RuntimeError, match="tuned route"):
        backend.prepare_linear(expected_weight, tokens=1, route="tuned")


def test_alignment_frozen_plan_and_capture_preparation(backend):
    from dataclasses import FrozenInstanceError

    weight = make_weight(80, 5120)
    plan = backend.prepare_linear(weight, tokens=8, route="tuned")
    raw = torch.empty(8 * 5120 + 1, dtype=torch.bfloat16, device="cuda")
    misaligned = raw[1:].view(8, 5120)
    out = plan.empty_output()
    with pytest.raises(ValueError, match="16-byte alignment"):
        plan.run(misaligned, out=out)
    with pytest.raises(FrozenInstanceError):
        plan.tokens = 9
    x = torch.zeros((8, 5120), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run(x, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run(x, out=out)
            with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
                backend.prepare_linear(weight, tokens=8)
    stream.synchronize()


def test_cuda_source_ownership_and_shared_readonly_plan(backend):
    cpu = make_weight(128, 128)
    weight = Q4RowSplitWeight(cpu.codes.cuda(), cpu.scales.cuda(), 128)
    plan = backend.prepare_linear(weight, tokens=2)
    weight.codes.zero_()
    weight.scales.zero_()
    x = torch.ones((2, 128), dtype=torch.bfloat16, device="cuda")
    outputs = [plan.empty_output(), plan.empty_output()]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, out in zip(streams, outputs):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan.run(x, out=out)
    for stream in streams:
        stream.synchronize()
    for out in outputs:
        assert_numerics(out, oracle(cpu, x), torch.bfloat16)
    alias = plan._weight.scales.view(torch.bfloat16).view(2, 128)
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=alias)
