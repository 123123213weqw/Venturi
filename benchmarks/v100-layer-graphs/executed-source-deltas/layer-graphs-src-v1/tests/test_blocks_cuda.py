"""Opt-in SM89 tests; builds are explicit and remote-only."""

from dataclasses import FrozenInstanceError
import os

import pytest
import torch

from block_checks import qualified_native_composition
from test_gated_mlp import make_weights
from venturi.models.gated_mlp import GatedMLPWorkspace
from venturi.ops import AdaBackend, CudaBackend, DenseWeight

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def backend():
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("set VENTURI_BLOCKS_LIBRARY to an explicitly built library")
    if not torch.cuda.is_available():
        pytest.fail("Native library selected but CUDA unavailable")
    return CudaBackend(library)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,features", [(1, 1), (3, 65), (17, 5120)])
def test_rmsnorm_guards_owned_weight_and_zero(backend, dtype, tokens, features):
    x = torch.randn(tokens, features, device="cuda", dtype=dtype)
    w = torch.linspace(0.5, 1.5, features)
    expected = x.cpu().double()
    expected = (
        expected
        * (expected.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        * w.double()
    ).to(dtype)
    plan = backend.prepare_rmsnorm(
        w, tokens=tokens, dtype=dtype, device="cuda:0", epsilon=1e-6
    )
    w.zero_()
    guard = torch.full((x.numel() + 2,), 47, device="cuda", dtype=dtype)
    out = guard[1:-1].view(tokens, features)
    plan.run(x, out=out)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.cpu(), expected, rtol=2 * torch.finfo(dtype).eps, atol=2e-6
    )
    assert guard[0].item() == guard[-1].item() == 47
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=x)
    x.zero_()
    plan.run(x, out=out)
    assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_pointwise_tail_inplace_extremes_and_partial_alias(backend, dtype):
    gate = torch.linspace(-90, 90, 259, device="cuda", dtype=dtype).view(7, 37)
    up = torch.linspace(-0.2, 0.2, 259, device="cuda", dtype=dtype).view(7, 37)
    original = gate.cpu().double()
    expected = (original * torch.sigmoid(original) * up.cpu().double()).to(dtype)
    plan = backend.prepare_swiglu(tokens=7, features=37, dtype=dtype, device="cuda:0")
    plan.run(gate, up, out=gate)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        gate.cpu(), expected, rtol=2 * torch.finfo(dtype).eps, atol=2e-6
    )
    add = backend.prepare_add(tokens=7, features=37, dtype=dtype, device="cuda:0")
    expected = (gate.cpu().double() + up.cpu().double()).to(dtype)
    add.run(gate, up, out=gate)
    torch.testing.assert_close(gate.cpu(), expected, rtol=0, atol=0)
    raw = torch.empty(260, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match="alias"):
        plan.run(raw[:-1].view(7, 37), up, out=raw[1:].view(7, 37))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,n,k", [(1, 3, 7), (7, 65, 129), (17, 128, 256)])
def test_dense_independent_fp64_and_shape_ownership(backend, dtype, tokens, n, k):
    weight = torch.randn(n, k, dtype=dtype) / k**0.5
    x = torch.randn(tokens, k, device="cuda", dtype=dtype)
    expected = (x.cpu().double() @ weight.double().T).to(dtype)
    plan = backend.prepare_linear(
        DenseWeight(weight), tokens=tokens, dtype=dtype, device="cuda:0"
    )
    weight.zero_()
    workspace = torch.empty(plan.workspace_bytes, device="cuda", dtype=torch.uint8)
    guard = torch.full((tokens * n + 2,), 47, device="cuda", dtype=dtype)
    out = guard[1:-1].view(tokens, n)
    plan.run(x, out=out, workspace=workspace)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.cpu(), expected, rtol=2 * torch.finfo(dtype).eps, atol=2e-5
    )
    assert guard[0].item() == guard[-1].item() == 47
    with pytest.raises(FrozenInstanceError):
        plan.tokens = tokens + 1
    with pytest.raises((ValueError, TypeError), match="workspace"):
        plan.run(x, out=out)
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.run(x, out=out, workspace=workspace)


@pytest.mark.parametrize("h,intermediate", [(32, 80), (40, 136), (65, 97)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_shared_block_fp64_graph_stream_and_allocations(
    backend, h, intermediate, dtype, monkeypatch
):
    weights = make_weights(h, intermediate, dtype)
    plan = weights.prepare(backend, tokens=7, dtype=dtype, device="cuda:0")
    ws = plan.empty_workspace()
    x = torch.randn(
        7, h, dtype=dtype, generator=torch.Generator().manual_seed(723 + h)
    ).cuda()
    out = torch.empty_like(x)
    expected = qualified_native_composition(plan, weights, x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run(x, out=out, workspace=ws)
    stream.synchronize()
    assert torch.equal(out, expected)
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run(x, out=out, workspace=ws)
        x.zero_()
        graph.replay()
    stream.synchronize()
    assert torch.count_nonzero(out).item() == 0
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    def forbidden(*args, **kwargs):
        raise AssertionError("allocating/fallback tensor helper called")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        patch.setattr(torch, "mm", forbidden)
        for _ in range(3):
            plan.run(x, out=out, workspace=ws)
    torch.cuda.synchronize()
    assert (
        torch.cuda.memory_allocated() == torch.cuda.max_memory_allocated() == allocated
    )
    bad = GatedMLPWorkspace(ws.hidden, ws.gate, ws.gate, ws.linear)
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=out, workspace=bad)
    with torch.cuda.stream(stream):
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            plan.run(x, out=out, workspace=ws)
            with pytest.raises(RuntimeError, match="outside CUDA graph"):
                plan.empty_workspace()


@pytest.mark.parametrize("tokens", [1, 7, 17])
def test_q4_composition_without_external_model_engine(backend, tokens):
    library = os.environ.get("VENTURI_ADA_LIBRARY")
    if not library:
        pytest.skip("set VENTURI_ADA_LIBRARY for explicit Q4 composition")
    native = CudaBackend(backend.library, q4_backend=AdaBackend(library))
    weights = make_weights(65, 97, torch.bfloat16, q4=True)
    plan = weights.prepare(native, tokens=tokens, dtype=torch.bfloat16, device="cuda:0")
    x = torch.randn(tokens, 65, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ws = plan.empty_workspace()
    assert ws.linear is None
    plan.run(x, out=out, workspace=ws)
    assert torch.equal(out, qualified_native_composition(plan, weights, x))


def test_qwen_sized_down_projection_near_cancellation_regression(backend):
    """Default cuBLAS compute failed the fixed bound on these represented inputs."""
    from venturi.models import GatedMLPWeights

    torch.set_num_threads(4)
    h, intermediate, tokens = 5120, 17408, 128
    generator = torch.Generator().manual_seed(812 + h + intermediate)

    def dense(n, k):
        return DenseWeight((torch.randn(n, k, generator=generator) / k**0.5).bfloat16())

    weights = GatedMLPWeights(
        torch.ones(h, dtype=torch.bfloat16),
        dense(intermediate, h),
        dense(intermediate, h),
        dense(h, intermediate),
    )
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261018)
        for features, count in [
            (4096, 1),
            (4096, 128),
            (4096, 1),
            (4096, 128),
            (5120, 1),
            (5120, 128),
        ]:
            x = (torch.randn(count, features, device="cuda") * 0.5).bfloat16()
    plan = weights.prepare(
        backend, tokens=tokens, dtype=torch.bfloat16, device="cuda:0"
    )
    ws = plan.empty_workspace()
    plan.norm.run(x, out=ws.hidden)
    plan.gate.run(ws.hidden, out=ws.gate, workspace=ws.linear)
    plan.up.run(ws.hidden, out=ws.up, workspace=ws.linear)
    plan.swiglu.run(ws.gate, ws.up, out=ws.gate)
    reference = ws.gate.cpu().double() @ weights.down.data.double().T
    plan.down.run(ws.gate, out=ws.hidden, workspace=ws.linear)
    error = (ws.hidden.cpu().double() - reference.bfloat16().double()).abs()
    bound = 2 * torch.finfo(torch.bfloat16).eps * reference.abs() + 2e-5
    assert bool((error <= bound).all()), float(error.max())


def test_fp16_whole_chain_counterexample_passes_per_op_bounds(backend):
    """Seed 7 previously failed a composite residual-relative assertion.

    Every Op must still meet the original fixed independent FP64 bound; replacing
    that with a larger whole-chain tolerance would hide actual operator defects.
    Exact composition and current-stream execution are checked separately.
    """
    weights = make_weights(40, 136, torch.float16)
    x = torch.randn(
        7, 40, dtype=torch.float16, generator=torch.Generator().manual_seed(7)
    ).cuda()
    plan = weights.prepare(backend, tokens=7, dtype=x.dtype, device="cuda:0")
    expected = qualified_native_composition(plan, weights, x)
    out = torch.empty_like(x)
    plan.run(x, out=out, workspace=plan.empty_workspace())
    assert torch.equal(out, expected)
    torch.cuda.synchronize()
    plan.close()
