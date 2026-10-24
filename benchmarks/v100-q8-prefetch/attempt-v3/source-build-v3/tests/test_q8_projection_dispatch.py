"""Opt-in native call chain versus the pinned qualified operator, before speed."""

import ctypes as C
import os

import pytest
import torch

from test_q8_prefetch import api, operands


@pytest.fixture(scope="module")
def dispatch(api):
    m, p, c = api
    p.venturi_q8_row_pair_enabled.restype = C.c_int
    assert p.venturi_q8_row_pair_enabled() == int(os.environ["VENTURI_EXPECT_ROW_PAIR"])
    p.venturi_q8_projection_fused_run.argtypes = p.venturi_q8_fused_run.argtypes
    p.venturi_q8_projection_fused_run.restype = C.c_int
    p.venturi_q8_projection_partial_run.argtypes = p.venturi_q8_column_partial_run.argtypes
    p.venturi_q8_projection_partial_run.restype = C.c_int
    return m, p, c


SHAPES = [
    (5120, [8704, 8704], False),
    (8704, [5120], True),
    (5120, [5120, 3072], False),
    (3072, [5120], True),
    (5120, [6144, 512, 512], False),
    (67, [33, 17], False),  # Unsupported shape always falls back.
    (1025, [33], True),
]


@pytest.mark.parametrize("tokens", [1, 2, 3, 7, 8, 9, 23, 32, 33])
@pytest.mark.parametrize("k,ns,partial", SHAPES)
def test_native_dispatch_bytes_and_fallbacks(dispatch, tokens, k, ns, partial):
    m, p, c = dispatch
    padded, _, weights, _, x, work = operands(k, ns, tokens)
    stream = torch.cuda.current_stream().cuda_stream
    m.check(p.venturi_q8_dp4a_quantize(x.data_ptr(), work.data_ptr(), work.numel(), tokens, k, 2, stream))
    outputs = []
    for lib, routed in ((c, False), (p, True)):
        dtype = torch.float32 if partial else torch.float16
        ys = [torch.empty(tokens, n, device="cuda", dtype=dtype) for n in ns]
        if partial:
            fn = lib.venturi_q8_projection_partial_run if routed else lib.venturi_q8_column_partial_run
            args = [work.data_ptr(), work.numel(), weights[0][0].data_ptr(), weights[0][1].data_ptr(), ys[0].data_ptr(), tokens, ns[0], k, padded]
        else:
            fn = lib.venturi_q8_projection_fused_run if routed else lib.venturi_q8_fused_run
            desc = (m.Descriptor * len(ns))(*[m.Descriptor(q.data_ptr(), s.data_ptr(), y.data_ptr(), n, n) for (q,s),y,n in zip(weights,ys,ns)])
            args = [work.data_ptr(), work.numel(), desc, len(ns), tokens, k, padded, 2]
        m.check(fn(*args, stream))
        outputs.append([y.cpu() for y in ys])
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in zip(*outputs))
