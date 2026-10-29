"""T1 warp graphs use fresh inputs, including signed zero and nonfinite data."""

import os

import pytest
import torch

from test_gdn import gdn_inputs
from test_gdn_compact import equal_bytes
from venturi.ops import CudaBackend
from venturi.ops.gdn_compact import prepare_gdn_compact


def test_fast_graph_dynamic_inputs_and_no_fallback(monkeypatch):
    candidate = os.environ.get("VENTURI_GDN_WARP_LIBRARY")
    control = os.environ.get("VENTURI_GDN_CONTROL_LIBRARY")
    if not candidate or not control:
        pytest.skip("explicit qualified warp-decode and control libraries required")
    values = list(gdn_inputs(tokens=1, hq=1, hv=3, dk=128, dv=7,
                             dtype=torch.float16, device="cuda:0"))
    for tensor in values:
        tensor.fill_(-0.0)
    plans = [prepare_gdn_compact(
        CudaBackend(library), tokens=1, qk_heads=1, value_heads=3,
        key_dim=128, value_dim=7, dtype=torch.float16, device="cuda:0",
    ) for library in (control, candidate)]
    outputs = [(torch.empty_like(values[2]), torch.empty_like(values[5])) for _ in plans]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("Warp GDN allocated or used a numerical fallback")

    with monkeypatch.context() as context:
        for name in ("empty", "empty_like", "matmul", "mm", "cat"):
            context.setattr(torch, name, forbidden)
        with torch.cuda.graph(graph, stream=stream):
            plans[1].run(*values[:5], state=values[5],
                         out=outputs[1][0], state_out=outputs[1][1])
    torch.cuda.current_stream().wait_stream(stream)
    for replacement in (-0.0, 0.25, float("nan"), 0.0):
        values[2].fill_(replacement)
        values[4].fill_(0.5)
        graph.replay()
        plans[0].run(*values[:5], state=values[5],
                     out=outputs[0][0], state_out=outputs[0][1])
        for old, new in zip(*outputs):
            assert torch.equal(torch.isnan(old), torch.isnan(new))
            assert torch.equal(torch.isinf(old), torch.isinf(new))
            finite = torch.isfinite(old)
            assert equal_bytes(old[finite], new[finite])
