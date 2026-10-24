"""T3 research schedules preserve packed inputs, output bytes and row guards."""

import os
import pytest
import torch
from test_q8_dp4a import setup
from q8_dp4a_reference import projection_reference
from q8_t3_helpers import plain, fused
from venturi.ops import CudaBackend, prepare_q8_dp4a
from venturi.ops.q8_fused import prepare_q8_fused


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_T3_LIBRARY")
    if not path:
        pytest.skip("explicit T3 research library required")
    return CudaBackend(path)


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("staging", [0, 1])
@pytest.mark.parametrize("rows", [4, 8, 16])
@pytest.mark.parametrize(
    "shape", [(7, 1), (33, 67), (257, 5120), (1025, 6145), (53, 17408)]
)
def test_plain_exact_reference_workspace_and_guards(
    backend, passes, staging, rows, shape
):
    codes, scales, x, r = setup(backend, *shape, 3)
    old = prepare_q8_dp4a(r, tokens=3, residual_compensation=passes == 2)
    if staging == 0 and old.workspace_bytes > 48 * 1024:
        with pytest.raises(ValueError, match="48KiB"):
            plain(r, rows, staging, passes)
        return
    new = plain(r, rows, staging, passes)
    work = [
        torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
        for p in (old, new)
    ]
    guards = [
        torch.full((3 * shape[0] + 2,), 47, dtype=torch.float16, device="cuda:0")
        for _ in range(2)
    ]
    for p, w, g in zip((old, new), work, guards):
        p.run(x.cuda(), out=g[1:-1].view(3, shape[0]), workspace=w)
    assert torch.equal(guards[0].view(torch.uint8), guards[1].view(torch.uint8))
    assert guards[1][0] == guards[1][-1] == 47 and torch.equal(*work)
    ref = projection_reference(x, codes, scales, passes)
    actual = guards[1][1:-1].view(3, shape[0]).cpu().double()
    assert (actual - ref).norm() / (ref.norm() + 1e-12) < 0.001


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("staging", [0, 1])
@pytest.mark.parametrize("rows", [4, 8, 16])
@pytest.mark.parametrize("ns", [(7, 33), (31, 17, 9)])
@pytest.mark.parametrize("k", [67, 5120, 17408])
def test_fused_exact_pitched_rows(backend, passes, staging, rows, ns, k):
    data = [setup(backend, n, k, 3) for n in ns]
    rs = [d[3] for d in data]
    if staging == 0 and 3 * ((k + 31) // 32) * passes * 36 > 48 * 1024:
        with pytest.raises(ValueError, match="48KiB"):
            fused(rs, rows, staging, passes)
        return
    strides = [n + 5 for n in ns]
    plans = [
        prepare_q8_fused(
            rs, tokens=3, residual_compensation=passes == 2, output_strides=strides
        ),
        fused(rs, rows, staging, passes, strides),
    ]
    work = torch.empty(plans[0].workspace_bytes, dtype=torch.uint8, device="cuda:0")
    outputs = [
        [torch.full((3, s), 47, dtype=torch.float16, device="cuda:0") for s in strides]
        for _ in plans
    ]
    x = data[0][2].cuda()
    for p, outs in zip(plans, outputs):
        p.run(x, outputs=outs, workspace=work)
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(*outputs)
    )
    assert all(bool((o[:, n:] == 47).all()) for o, n in zip(outputs[1], ns))


@pytest.mark.parametrize("staging", [0, 1])
def test_graph_nonfinite_alias_and_invalid_geometry(backend, staging):
    _, _, x, r = setup(backend, 33, 128, 3)
    p = plain(r, 8, staging)
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    out = torch.empty(3, 33, dtype=torch.float16, device="cuda:0")
    x = x.cuda()
    p.run(x, out=out, workspace=work)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        p.run(x, out=out, workspace=work)
    torch.cuda.current_stream().wait_stream(stream)
    x.fill_(float("inf"))
    g.replay()
    assert bool(torch.isnan(out).all())
    x.zero_()
    g.replay()
    assert not out.any()
    with pytest.raises(ValueError, match="alias"):
        p.run(x, out=work[:198].view(torch.float16).view(3, 33), workspace=work)
    args = [
        work.data_ptr(),
        work.numel(),
        p.base.resident._codes.data_ptr(),
        p.base.resident._scales.data_ptr(),
        out.data_ptr(),
        3,
        33,
        128,
        128,
        2,
        8,
        staging,
        torch.cuda.current_stream().cuda_stream,
    ]
    for idx, v in [
        (0, 0),
        (1, work.numel() - 1),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 2),
        (6, 0),
        (7, 0),
        (8, 1),
        (9, 3),
        (10, 1),
        (11, 2),
        (0, work.data_ptr() + 1),
    ]:
        bad = args.copy()
        bad[idx] = v
        assert p._run(*bad) == -1
    torch.cuda.synchronize()
