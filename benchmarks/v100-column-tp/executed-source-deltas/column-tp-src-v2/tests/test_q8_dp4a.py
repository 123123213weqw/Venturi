"""New activation arithmetic is opt-in and separately checked, never byte-parity claimed."""

import os

import pytest
import torch

from q8_dp4a_reference import quantize_reference, projection_reference
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a


def test_independent_profile_bounds_ties_and_compensation():
    x = torch.randn(3, 67, generator=torch.Generator().manual_seed(293)).half()
    for passes in (1, 2):
        q, d, represented = quantize_reference(x, passes)
        assert q.min() >= -127 and q.max() <= 127
        assert not q[:, :, -1, 3:].any()
        errors = (represented - x.double()).abs()
        bound = d[-1].double().repeat_interleave(32, -1)[:, :67] * 0.5001
        assert (errors <= bound + 2e-7 * x.double().abs()).all()
    _, _, a = quantize_reference(x, 1)
    _, _, b = quantize_reference(x, 2)
    assert (b - x.double()).norm() < (a - x.double()).norm() / 100
    ties = torch.zeros(1, 32, dtype=torch.float16)
    ties[0, :5] = torch.tensor([127, 0.5, 1.5, -0.5, -1.5])
    q, d, _ = quantize_reference(ties, 1)
    assert d.item() == 1 and q.flatten()[:5].tolist() == [127, 0, 2, 0, -2]
    with pytest.raises(ValueError, match="resident native Q8"):
        prepare_q8_dp4a(object(), tokens=1, residual_compensation=True)


@pytest.fixture
def backend():
    library = os.environ.get("VENTURI_DP4A_LIBRARY")
    if not library:
        pytest.skip("explicit experimental DP4A library required")
    return CudaBackend(library)


def setup(backend, n, k, tokens):
    padded = (k + 127) // 128 * 128
    gen = torch.Generator().manual_seed(5821 + k + n + tokens)
    codes = torch.randint(-127, 128, (n, padded), dtype=torch.int8, generator=gen)
    scales = (torch.rand(n, padded // 32, generator=gen) * 0.01).half()
    scales[:, 0] = 0
    x = torch.randn(tokens, k, generator=gen).half()
    r = backend.upload_linear(
        Q8RowSplitWeight(codes, scales, k), dtype=torch.float16, device="cuda:0"
    )
    return codes, scales, x, r


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 23, 32])
@pytest.mark.parametrize("shape", [(7, 1), (33, 67), (17, 128), (9, 129), (257, 5120)])
def test_profile_quantizer_packing_integer_dot_and_tail_guards(
    backend, passes, tokens, shape
):
    codes, scales, x, r = setup(backend, *shape, tokens)
    plan = prepare_q8_dp4a(r, tokens=tokens, residual_compensation=passes == 2)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    guard = torch.full(
        (tokens * shape[0] + 2,), 47, dtype=torch.float16, device="cuda:0"
    )
    out = guard[1:-1].view(tokens, shape[0])
    plan.run(x.cuda(), out=out, workspace=workspace)
    actual = out.cpu()
    assert guard[0] == guard[-1] == 47
    assert bool(torch.isfinite(actual).all())
    q, d, _ = quantize_reference(x, passes)
    size = q.numel()
    assert torch.equal(workspace[:size].cpu().view(torch.int8).reshape(q.shape), q)
    assert torch.equal(workspace[size:].cpu().view(torch.float32).reshape(d.shape), d)
    ideal = projection_reference(x, codes, scales, passes)
    rms = ideal.square().mean().sqrt().clamp_min(1e-8)
    assert (
        torch.linalg.vector_norm(actual.double() - ideal) / (ideal.norm() + 1e-12)
        < 0.001
    )
    assert (
        (actual.double() - ideal.half().double()).abs()
        <= 2 * torch.finfo(torch.float16).eps * torch.maximum(ideal.abs(), rms)
    ).all()
    assert plan.precision_profile.endswith("-residual-a8") == (passes == 2)


def test_explicit_workspace_alias_graph_no_torch_and_nonfinite(backend, monkeypatch):
    _, _, x, r = setup(backend, 16, 128, 1)
    plan = prepare_q8_dp4a(r, tokens=1, residual_compensation=True)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    x = x.cuda()
    out = torch.empty(1, 16, dtype=torch.float16, device="cuda:0")
    for bad in (None, workspace[:-1]):
        with pytest.raises((ValueError, TypeError)):
            plan.run(x, out=out, workspace=bad)
    with pytest.raises(ValueError, match="alias"):
        plan.run(
            x, out=workspace[:32].view(torch.float16).view(1, 16), workspace=workspace
        )
    unaligned = torch.empty(
        plan.workspace_bytes + 16, dtype=torch.uint8, device="cuda:0"
    )
    with pytest.raises(ValueError, match="16-byte alignment"):
        plan.run(x, out=out, workspace=unaligned[4 : 4 + plan.workspace_bytes])
    with pytest.raises(ValueError, match="boolean"):
        prepare_q8_dp4a(r, tokens=1, residual_compensation="yes")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run(x, out=out, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    expected = out.cpu().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("DP4A execution called Torch math/allocation")

    with monkeypatch.context() as p:
        for name in ("empty", "zeros", "mm", "matmul", "round"):
            p.setattr(torch, name, forbidden)
        plan.run(x, out=out, workspace=workspace)
    assert torch.equal(out.cpu(), expected)
    x.fill_(float("inf"))
    plan.run(x, out=out, workspace=workspace)
    assert not torch.isfinite(out).any()
    x.zero_()
    plan.run(x, out=out, workspace=workspace)
    assert not out.any()
    assert not workspace.any()
