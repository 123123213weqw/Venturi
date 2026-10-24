"""SIMT Q8 tiles must preserve the original dot-product and reduction bytes."""

import os

import pytest
import torch

from test_fp32_parameters import mixed_port as base_port
from venturi.ops import CudaBackend, Q8RowSplitWeight


@pytest.fixture(params=["native-fp16", "native-bf16"])
def native_engine(request):
    return base_port.__wrapped__(request)


@pytest.mark.parametrize(
    "shape", [(1, 1), (7, 65), (33, 129), (7, 256), (33, 512), (257, 5120)]
)
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 9, 26])
def test_q8_tiled_original_bytes_tails_and_output_guards(native_engine, shape, tokens):
    path = os.environ.get("VENTURI_Q8_BASELINE_LIBRARY")
    if not path:
        pytest.skip("explicit original Q8 baseline library required")
    backend, dtype, device = native_engine
    baseline = CudaBackend(path)
    n, k = shape
    padded = (k + 127) // 128 * 128
    generator = torch.Generator().manual_seed(9367 + n + tokens)
    codes = torch.randint(-127, 128, (n, padded), dtype=torch.int8, generator=generator)
    scales = (torch.rand(n, padded // 32, generator=generator) * 0.004).half()
    if k > 32:
        scales[:, 0] = 0
    weight = Q8RowSplitWeight(codes, scales, k)
    x = torch.randn(tokens, k, generator=generator).to(dtype=dtype, device=device)
    outputs = []
    for engine in (baseline, backend):
        resident = engine.upload_linear(weight, dtype=dtype, device=device)
        guard = torch.full((tokens * n + 2,), 47, dtype=dtype, device=device)
        resident.prepare(tokens=tokens).run(x, out=guard[1:-1].reshape(tokens, n))
        outputs.append(guard.cpu())
        assert outputs[-1][0] == outputs[-1][-1] == 47
    assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
