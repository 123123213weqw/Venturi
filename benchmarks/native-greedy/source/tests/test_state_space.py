"""Independent staged FP64 formulas for the missing GDN frontend operators."""

import os

import pytest
import torch

from venturi.ops import CudaBackend, TorchBackend


@pytest.fixture(params=["fp64", "float16", "bfloat16", "float32"])
def engine(request):
    if request.param == "fp64":
        return TorchBackend(), torch.float64, torch.device("cpu")
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    return CudaBackend(library), getattr(torch, request.param), torch.device("cuda:0")


def sample(shape, dtype, device, seed=38):
    return (
        torch.randn(
            shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
        )
        * 0.3
    ).to(dtype=dtype, device=device)


def silu64(x):
    return x * x.sigmoid()


def conv_oracle(x, weight, history):
    dtype = x.dtype
    x, weight, history = (t.cpu().double() for t in (x, weight, history))
    joined = torch.cat((history, x))
    out = torch.empty_like(x)
    for token in range(x.shape[0]):
        # Independent window-by-window reduction, not the production tap loop.
        accum = (joined[token : token + weight.shape[0]] * weight).sum(0)
        out[token] = silu64(accum.to(dtype).double())
    return out, joined[-history.shape[0] :].to(dtype) if history.shape[
        0
    ] else history.to(dtype)


def assert_qualified(actual, ideal, floor=2e-5):
    dtype = actual.dtype
    ideal = ideal.cpu().double()
    error = (actual.cpu().double() - ideal.to(dtype).double()).abs()
    assert bool((error <= 2 * torch.finfo(dtype).eps * ideal.abs() + floor).all()), (
        error.max().item(),
        ideal.abs().max().item(),
    )


@pytest.mark.parametrize("kernel", [1, 2, 4, 16])
def test_conv_prefill_chunks_inplace_history_and_shared_plans(engine, kernel):
    backend, dtype, device = engine
    x = sample((13, 37), dtype, device)
    weight = sample((kernel, 37), dtype, "cpu", 12)
    history = sample((kernel - 1, 37), dtype, device, 11)
    resident = backend.upload_causal_conv(weight, dtype=dtype, device=device)
    plan = resident.prepare(tokens=13)
    pointers = [t.data_ptr() for t in resident.storage_tensors]
    expected, expected_history = conv_oracle(x, weight, history)
    out, final = torch.empty_like(x), torch.empty_like(history)
    plan.run(x, history=history, out=out, history_out=final)
    assert_qualified(out, expected)
    assert torch.equal(final.cpu(), expected_history)
    chunks, current, start = [], history.clone(), 0
    for count in (1, 2, 7, 3):
        chunk = resident.prepare(tokens=count)
        target = torch.empty_like(x[start : start + count])
        chunk.run(
            x[start : start + count], history=current, out=target, history_out=current
        )
        chunks.append(target)
        start += count
    if dtype == torch.float64:
        # Shape-dependent reference elementwise dispatch can differ by a few
        # FP64 ulps; this is not a license to change native/per-op bounds.
        torch.testing.assert_close(torch.cat(chunks), out, rtol=2e-14, atol=2e-14)
    else:
        assert torch.equal(torch.cat(chunks), out)
    assert torch.equal(current, final)
    assert pointers == [t.data_ptr() for t in resident.storage_tensors]


@pytest.mark.parametrize("round_beta", [True, False])
def test_gate_transforms_fp64_and_extremes(engine, round_beta):
    backend, dtype, device = engine
    a = sample((9, 7), dtype, device)
    b = sample((9, 7), dtype, device, 23)
    a[0, :4] = torch.tensor([-100, -20, 20, 100], dtype=dtype, device=device)
    b[0, :4] = torch.tensor([-100, -20, 20, 100], dtype=dtype, device=device)
    log = sample((7,), torch.float64, "cpu", 20)
    bias = sample((7,), torch.float64, "cpu", 21)
    resident = backend.upload_gdn_gates(log, bias, dtype=dtype, device=device)
    work = torch.float64 if dtype == torch.float64 else torch.float32
    g = torch.empty(9, 7, dtype=work, device=device)
    beta = torch.empty_like(g)
    resident.prepare(tokens=9, round_beta=round_beta).run(a, b, log_decay=g, beta=beta)
    # Parameters are represented in their declared gate arithmetic dtype before
    # exponentiation. Exp is a prepared constant, independently observed here.
    rate = resident._rate.cpu().double()
    z = a.cpu().double() + resident._bias.cpu().double()
    expected_g = -rate * torch.where(z > 20, z, torch.log1p(z.exp()))
    expected_beta = b.cpu().double().sigmoid()
    if round_beta:
        expected_beta = expected_beta.to(dtype).double()
    torch.testing.assert_close(
        g.cpu().double(),
        expected_g,
        rtol=2e-6 if work == torch.float32 else 2e-14,
        atol=2e-7 if work == torch.float32 else 2e-14,
    )
    torch.testing.assert_close(
        beta.cpu().double(),
        expected_beta,
        rtol=2e-6 if work == torch.float32 else 2e-14,
        atol=2e-7 if work == torch.float32 else 2e-14,
    )
    assert bool((g <= 0).all()) and bool(((beta >= 0) & (beta <= 1)).all())


@pytest.mark.parametrize("features", [13, 128, 768])
def test_staged_gated_rmsnorm_and_zero_centered_norm(engine, features):
    backend, dtype, device = engine
    x, z = (
        sample((5, 3, features), dtype, device),
        sample((5, 3, features), dtype, device, 14),
    )
    gamma = sample((features,), dtype, "cpu", 8) + 1
    resident = backend.upload_gated_rmsnorm(gamma, dtype=dtype, device=device)
    pointers = [t.data_ptr() for t in resident.storage_tensors]
    out = torch.empty_like(x)
    resident.prepare(tokens=5, heads=3).run(x, z, out=out)
    work = x.cpu().double()
    normalized = (
        (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6))
        .to(dtype)
        .double()
    )
    weighted = (normalized * gamma.double()).to(dtype).double()
    expected = weighted * silu64(z.cpu().double())
    assert_qualified(out, expected)
    assert pointers == [t.data_ptr() for t in resident.storage_tensors]
    raw = sample((features,), dtype, "cpu", 19)
    before = raw.clone()
    norm = backend.upload_zero_centered_rmsnorm(raw, dtype=dtype, device=device)
    flat = x.reshape(-1, features)
    result = torch.empty_like(flat)
    norm.prepare(tokens=15).run(flat, out=result)
    expected = flat.cpu().double()
    expected = (
        expected
        * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + raw.double())
    )
    assert_qualified(result, expected)
    assert torch.equal(raw, before)


def test_validation_aliases_and_failed_preparation(engine):
    backend, dtype, device = engine
    x = sample((3, 13), dtype, device)
    resident = backend.upload_causal_conv(
        sample((4, 13), dtype, "cpu"), dtype=dtype, device=device
    )
    history, out = torch.empty_like(x), torch.full_like(x, 47)
    with pytest.raises(ValueError, match="alias"):
        resident.prepare(tokens=3).run(x, history=history, out=x, history_out=history)
    with pytest.raises(ValueError, match="dtype"):
        resident.prepare(tokens=3).run(
            x, history=history, out=out, history_out=history.to(torch.int32)
        )
    assert bool((out == 47).all())
    with pytest.raises(ValueError, match="kernel"):
        backend.upload_causal_conv(
            sample((33, 13), dtype, "cpu"), dtype=dtype, device=device
        )
    with pytest.raises(ValueError, match="finite"):
        backend.upload_gdn_gates(
            torch.tensor([1000.0]), torch.tensor([0.0]), dtype=dtype, device=device
        )
    r = backend.upload_gdn_gates(
        torch.zeros(13), torch.zeros(13), dtype=dtype, device=device
    )
    work = torch.float64 if dtype == torch.float64 else torch.float32
    gates = torch.empty(3, 13, dtype=work, device=device)
    with pytest.raises(ValueError, match="alias"):
        r.prepare(tokens=3).run(x, x, log_decay=gates, beta=gates)
    norm = backend.upload_gated_rmsnorm(
        torch.ones(13, dtype=dtype), dtype=dtype, device=device
    )
    x3 = x.reshape(3, 1, 13)
    with pytest.raises(ValueError, match="alias"):
        norm.prepare(tokens=3, heads=1).run(x3, x3, out=x3)


@pytest.mark.cuda
def test_new_native_ops_capture_and_no_numerical_fallback(monkeypatch):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    backend, dtype, device = CudaBackend(library), torch.bfloat16, "cuda:0"
    x = sample((7, 13), dtype, device)
    conv = backend.upload_causal_conv(
        sample((4, 13), dtype, "cpu"), dtype=dtype, device=device
    ).prepare(tokens=7)
    history, final, conv_out = (
        torch.zeros(3, 13, dtype=dtype, device=device),
        torch.empty(3, 13, dtype=dtype, device=device),
        torch.empty_like(x),
    )
    gates = backend.upload_gdn_gates(
        torch.zeros(13), torch.zeros(13), dtype=dtype, device=device
    ).prepare(tokens=7)
    g, beta = (torch.empty_like(x, dtype=torch.float32) for _ in range(2))
    norm = backend.upload_gated_rmsnorm(
        torch.ones(13, dtype=dtype), dtype=dtype, device=device
    ).prepare(tokens=7, heads=1)
    z = sample((7, 1, 13), dtype, device, 99)
    out = torch.empty_like(z)

    def execute():
        conv.run(x, history=history, out=conv_out, history_out=final)
        gates.run(x, conv_out, log_decay=g, beta=beta)
        norm.run(conv_out.view(7, 1, 13), z, out=out)

    execute()
    expected = [t.clone() for t in (conv_out, final, g, beta, out)]
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())

    def forbidden(*args, **kwargs):
        raise AssertionError("Native inference allocated or used numerical fallback")

    with torch.cuda.stream(stream), monkeypatch.context() as m:
        for name in (
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "cat",
            "mm",
            "matmul",
        ):
            m.setattr(torch, name, forbidden)
        for name in ("to", "clone", "copy_", "float", "exp", "sigmoid"):
            m.setattr(torch.Tensor, name, forbidden)
        with torch.cuda.graph(graph, stream=stream):
            execute()
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert all(
        torch.equal(actual, reference)
        for actual, reference in zip((conv_out, final, g, beta, out), expected)
    )


def test_fp32_gated_gamma_preserves_promoted_product_and_source_precision(engine):
    backend, dtype, device = engine
    x = sample((7, 3, 128), dtype, device, 401)
    z = sample((7, 3, 128), dtype, device, 402)
    weight = (torch.linspace(0.90013, 1.10177, 128)).float()
    source = weight.clone()
    resident = backend.upload_gated_rmsnorm(weight, dtype=dtype, device=device)
    plan = resident.prepare(tokens=7, heads=3)
    out = torch.empty_like(x)
    plan.run(x, z, out=out)
    values = x.cpu().double()
    normal = (
        (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6))
        .to(dtype)
        .double()
    )
    ideal = normal * weight.double() * silu64(z.cpu().double())
    assert_qualified(out, ideal)
    assert not resident.round_weighted
    assert resident._weight.dtype == (
        torch.float64 if dtype == torch.float64 else torch.float32
    )
    assert torch.equal(weight, source)
    if dtype in (torch.float16, torch.bfloat16):
        assert not torch.equal(resident._weight.cpu(), weight.to(dtype).float())


def test_original_fp32_gated_gamma_matches_eager_qwen_source():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    x = sample((7, 3, 128), torch.bfloat16, "cpu", 510)
    z = sample((7, 3, 128), torch.bfloat16, "cpu", 511)
    weight = torch.linspace(0.90013, 1.10177, 128, dtype=torch.float32)
    hf = Qwen3_5RMSNormGated(128)
    with torch.no_grad():
        hf.weight.copy_(weight)
    expected = hf(x, z)
    plan = (
        TorchBackend()
        .upload_gated_rmsnorm(weight, dtype=torch.bfloat16, device="cpu")
        .prepare(tokens=7, heads=3)
    )
    out = torch.empty_like(x)
    plan.run(x, z, out=out)
    assert torch.equal(out, expected)
