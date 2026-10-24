"""Shared activation transport preserves quantizer/reduction bits and row guards."""

import os
import pytest
import torch
from test_q8_dp4a import setup
from q8_dp4a_reference import projection_reference
from venturi.ops import CudaBackend, prepare_q8_dp4a
from venturi.ops.q8_shared import prepare_q8_tiled_shared


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_Q8_TILED_SHARED_LIBRARY")
    if not path:
        pytest.skip("explicit experimental shared-activation library required")
    return CudaBackend(path)


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("tokens", [3, 4, 8, 23, 32])
@pytest.mark.parametrize("shape", [(7, 67), (257, 5120), (1025, 6145)])
@pytest.mark.parametrize("rows", [4, 8, 16])
@pytest.mark.parametrize("tile", [4, 8])
def test_exact_original_bytes_workspace_tails_and_fp64(
    backend, passes, tokens, shape, rows, tile
):
    codes, scales, x, r = setup(backend, *shape, tokens)
    old = prepare_q8_dp4a(r, tokens=tokens, residual_compensation=passes == 2)
    new = prepare_q8_tiled_shared(
        r, tokens=tokens, rows=rows, token_tile=tile, residual_compensation=passes == 2
    )
    scratch = [
        torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
        for p in (old, new)
    ]
    guards = [
        torch.full((tokens * shape[0] + 2,), 47, dtype=torch.float16, device="cuda:0")
        for _ in range(2)
    ]
    x_gpu = x.cuda()
    for p, work, guard in zip((old, new), scratch, guards):
        p.run(x_gpu, out=guard[1:-1].view(tokens, shape[0]), workspace=work)
    assert torch.equal(guards[0].view(torch.uint8), guards[1].view(torch.uint8))
    assert guards[1][0] == guards[1][-1] == 47
    assert torch.equal(scratch[0], scratch[1])
    ideal = projection_reference(x, codes, scales, passes)
    actual = guards[1][1:-1].view(tokens, shape[0]).cpu().double()
    assert (actual - ideal).norm() / (ideal.norm() + 1e-12) < 0.001


def test_graph_invalid_geometry_and_workspace_alias(backend):
    _, _, x, r = setup(backend, 33, 128, 5)
    p = prepare_q8_tiled_shared(
        r, tokens=5, rows=8, token_tile=8, residual_compensation=True
    )
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    out = torch.empty(5, 33, dtype=torch.float16, device="cuda:0")
    x = x.cuda()
    p.run(x, out=out, workspace=work)
    expected = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        p.run(x, out=out, workspace=work)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    with pytest.raises(ValueError, match="alias"):
        p.run(x, out=work[:330].view(torch.float16).view(5, 33), workspace=work)
    args = [
        work.data_ptr(),
        work.numel(),
        p.base.resident._codes.data_ptr(),
        p.base.resident._scales.data_ptr(),
        out.data_ptr(),
        5,
        33,
        128,
        128,
        2,
        8,
        8,
        torch.cuda.current_stream().cuda_stream,
    ]
    for idx, value in [
        (0, 0),
        (1, work.numel() - 1),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 1),
        (9, 3),
        (10, 1),
        (11, 3),
        (0, work.data_ptr() + 1),
        (2, p.base.resident._codes.data_ptr() + 1),
    ]:
        invalid = args.copy()
        invalid[idx] = value
        assert p._run(*invalid) == -1
    torch.cuda.synchronize()
    x.fill_(float("inf"))
    graph.replay()
    assert torch.isnan(out).all()
    x.zero_()
    graph.replay()
    assert not out.any()


def test_cpu_preflight():
    for tokens, rows, tile in [
        (0, 4, 4),
        (2, 4, 4),
        (True, 4, 4),
        (8, 3, 4),
        (8, True, 4),
        (8, 4, 3),
        (8, 4, True),
    ]:
        with pytest.raises(ValueError, match="K-tiled"):
            prepare_q8_tiled_shared(
                object(),
                tokens=tokens,
                rows=rows,
                token_tile=tile,
                residual_compensation=True,
            )
