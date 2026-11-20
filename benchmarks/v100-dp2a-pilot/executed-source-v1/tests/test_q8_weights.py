"""Explicit Q8 source quantization, generic projections and full block composition."""

from dataclasses import replace
import os

import pytest
import torch

from block_checks import qualified_native_composition
from block_reference import decode, gated_mlp
from decoder_checks import qualified_decoder_composition
from test_decoder_layer import layer_weights, rope_tables
from test_gated_mlp import make_weights
from venturi.models import AttentionDimensions
from venturi.ops import CudaBackend, Q4RowSplitWeight, Q8RowSplitWeight, TorchBackend


@pytest.mark.parametrize("k", [1, 31, 32, 65, 128, 129])
def test_explicit_quantization_nearest_even_stored_scale_and_padding(k):
    source = (
        torch.randn(
            11, k, generator=torch.Generator().manual_seed(38), dtype=torch.bfloat16
        )
        * 0.01
    )
    source[0].zero_()
    saved = source.clone()
    a = Q8RowSplitWeight.from_dense(source, rows_per_chunk=3)
    b = Q8RowSplitWeight.from_dense(source, rows_per_chunk=11)
    assert torch.equal(a.codes, b.codes) and torch.equal(a.scales, b.scales)
    assert torch.equal(source, saved)
    assert a.format == "q8_g32_fp16" and a.group_size == 32
    assert a.nbytes == a.codes.numel() + 2 * a.scales.numel()
    assert not bool((a.codes == -128).any())
    assert torch.count_nonzero(a.codes[:, k:]) == 0
    assert torch.count_nonzero(a.scales[0]) == 0
    exact = decode(a)
    assert torch.equal(exact, a.dequantize_reference())
    # Independently scalar-quantize each real group using stored scale and Python
    # nearest-even rounding, including partial groups and FP16 scale rounding.
    for row in range(11):
        for begin in range(0, k, 32):
            group = source[row, begin : begin + 32].float()
            stored = torch.tensor(
                float(group.abs().max()) / 127, dtype=torch.float16
            ).item()
            for col, value in enumerate(group.tolist(), begin):
                code = 0 if stored == 0 else max(-127, min(127, round(value / stored)))
                assert a.codes[row, col].item() == code
    error = (source.double() - exact).abs()
    scale = a.scales.double().repeat_interleave(32, -1)[:, :k]
    assert bool((error <= scale * 0.501 + 1e-12).all())


def test_explicit_underflow_overflow_nonfinite_and_no_q4_upgrade():
    source = torch.full((2, 32), 1e-12)
    result = Q8RowSplitWeight.from_dense(source)
    assert torch.count_nonzero(result.codes) == torch.count_nonzero(result.scales) == 0
    for value in (float("inf"), float("nan"), 1e12):
        with pytest.raises(ValueError):
            Q8RowSplitWeight.from_dense(torch.full((2, 32), value))
    with pytest.raises(TypeError):
        Q8RowSplitWeight.from_dense(
            Q4RowSplitWeight.from_codes(
                torch.zeros(2, 32, dtype=torch.int8),
                torch.ones(2, 1, dtype=torch.float16),
            )
        )
    with pytest.raises(ValueError, match="rows_per_chunk"):
        Q8RowSplitWeight.from_dense(torch.ones(2, 32), rows_per_chunk=0)


@pytest.mark.parametrize("bad", ["codes", "scales", "shape", "noncontiguous"])
def test_descriptor_guards(bad):
    w = Q8RowSplitWeight.from_dense(torch.randn(3, 65))
    kwargs = dict(codes=w.codes, scales=w.scales, in_features=w.in_features)
    if bad == "codes":
        kwargs["codes"] = w.codes.to(torch.uint8)
    elif bad == "scales":
        kwargs["scales"] = w.scales.float()
    elif bad == "shape":
        kwargs["in_features"] = 129
    else:
        kwargs["codes"] = torch.empty(3, 256, dtype=torch.int8)[:, ::2]
    with pytest.raises(ValueError):
        Q8RowSplitWeight(**kwargs)


@pytest.fixture(params=["reference", "cuda"])
def engine(request):
    if request.param == "reference":
        return TorchBackend(), torch.float64, "cpu"
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    return CudaBackend(library), torch.bfloat16, "cuda:0"


@pytest.mark.parametrize("tokens", [1, 7, 17])
def test_generic_q8_linear_and_shared_weight_plans(engine, tokens):
    backend, dtype, device = engine
    weight = Q8RowSplitWeight.from_dense(
        torch.randn(31, 65, generator=torch.Generator().manual_seed(18)) * 0.01
    )
    resident = backend.upload_linear(weight, dtype=dtype, device=device)
    pointers = tuple(t.data_ptr() for t in resident.storage_tensors)
    plan = resident.prepare(tokens=tokens)
    x = torch.randn(tokens, 65, dtype=dtype, device=device)
    out = torch.empty(tokens, 31, dtype=dtype, device=device)
    plan.run(x, out=out)
    expected = x.cpu().double() @ decode(weight).T
    if dtype == torch.float64:
        torch.testing.assert_close(out, expected, atol=1e-14, rtol=1e-12)
    else:
        assert bool(
            (
                (out.cpu().double() - expected.bfloat16().double()).abs()
                <= 2 * torch.finfo(dtype).eps * expected.abs() + 2e-4
            ).all()
        )
    assert "q8" in plan.route
    assert resident.prepare(tokens=1).storage_tensors[0].data_ptr() == pointers[0]
    lookup = resident.prepare_embedding([1, 30])
    embeddings = torch.empty(2, 65, dtype=dtype, device=device)
    lookup.run(out=embeddings)
    assert torch.equal(embeddings.cpu(), decode(weight)[[1, 30]].to(dtype))
    if dtype != torch.float64:
        assert lookup.resident._codes.data_ptr() == pointers[0]


def test_q8_mlp_and_decoder_composition(engine):
    backend, dtype, device = engine
    original = make_weights(32, 80, dtype=dtype)

    def quant(w):
        return Q8RowSplitWeight.from_dense(w.data.float())

    weights = replace(
        original,
        gate=quant(original.gate),
        up=quant(original.up),
        down=quant(original.down),
    )
    plan = weights.prepare(backend, tokens=7, dtype=dtype, device=device)
    x = torch.randn(7, 32, dtype=dtype, device=device)
    out, workspace = torch.empty_like(x), plan.empty_workspace()
    expected = (
        gated_mlp(weights, x, 1e-6)
        if dtype == torch.float64
        else qualified_native_composition(plan, weights, x)
    )
    plan.run(x, out=out, workspace=workspace)
    if dtype == torch.float64:
        torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
    else:
        assert torch.equal(out, expected)
    dims = AttentionDimensions(2, 1, 64)
    original = layer_weights(32, 80, dims, dtype)
    weights = replace(
        original,
        query=quant(original.query),
        key=quant(original.key),
        value=quant(original.value),
        output=quant(original.output),
        mlp=weights,
    )
    resident = weights.upload(backend, dimensions=dims, dtype=dtype, device=device)
    decoder = resident.prepare(tokens=7)
    cos, sin = rope_tables(7, dims, device)
    if dtype == torch.float64:
        from decoder_reference import decoder_reference

        expected, _ = decoder_reference(weights, dims, x, cos, sin)
        checks = {}
    else:
        expected, _, checks = qualified_decoder_composition(
            decoder.at(position=0), weights, x, cos, sin
        )
    decoder.at(position=0).run(
        x,
        cos,
        sin,
        out=out,
        workspace=decoder.empty_workspace(),
        cache=resident.empty_cache(capacity=7),
    )
    if dtype == torch.float64:
        torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
    else:
        assert torch.equal(out, expected)
        assert all(c["passed"] for c in checks.values())
