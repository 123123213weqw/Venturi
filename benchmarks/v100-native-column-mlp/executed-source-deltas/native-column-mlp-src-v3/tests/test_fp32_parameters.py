"""Lossless small parameters with FP16 activations; no silent half-weight casts."""

import pytest
import torch

from test_fp16_port import port as base_port
from test_state_space import conv_oracle, assert_qualified
from test_qwen_source import fixture_tensors, source_config, write_source
from test_qwen_q8_layer_tool import load as load_tool
from venturi.models import ResidentQwenGDNLayer, QwenQ8Weights, qwen_resident_bytes
from venturi.ninfer import NInferArtifact
from venturi.ops import DenseWeight, TorchBackend, CudaBackend
from venturi.qwen_export import export_qwen_q8


@pytest.fixture(params=["cpu-fp16", "native-fp16", "cpu-bf16", "native-bf16"])
def mixed_port(request):
    import os

    if request.param.endswith("bf16"):
        if request.param.startswith("cpu"):
            return TorchBackend(), torch.bfloat16, "cpu"
        if os.environ.get("VENTURI_SM70_LIBRARY"):
            pytest.skip("Native BF16 is not supported on SM70")
        library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
        if not library:
            pytest.skip("explicit SM89 native library required")
        return CudaBackend(library), torch.bfloat16, "cuda:0"
    return base_port.__wrapped__(request)


def test_tiny_bf16_linear_parameters_survive_and_affect_output(mixed_port):
    backend, dtype, device = mixed_port
    # This BF16 number vanishes when cast to FP16. Multiplication before the
    # output cast has a visible FP16 result, so retaining bytes alone is not enough.
    source = torch.tensor([[1e-8, 0, 0], [-1e-8, 0, 0]], dtype=torch.bfloat16)
    assert not bool(source.half().any())
    descriptor = DenseWeight(source.float())
    resident = backend.upload_fp32_linear(descriptor, dtype=dtype, device=device)
    assert torch.equal(resident.storage_tensors[0].cpu(), source.float())
    assert (
        resident.dtype == dtype and resident.storage_tensors[0].dtype == torch.float32
    )
    plan = resident.prepare(tokens=2)
    x = torch.tensor([[65504, 0, 0], [-65504, 0, 0]], dtype=dtype, device=device)
    out = torch.empty(2, 2, dtype=dtype, device=device)
    plan.run(x, out=out)
    expected = (x.cpu().double() @ source.double().T).to(dtype)
    assert bool(expected.abs().min() > 0) and torch.equal(out.cpu(), expected)
    with pytest.raises(ValueError, match="typed embedding"):
        resident.prepare_embedding([0])
    with pytest.raises(ValueError, match="no workspace"):
        plan.run(x, out=out, workspace=torch.empty(1, dtype=torch.uint8, device=device))
    for bad in (DenseWeight(source.half()), DenseWeight(source.double())):
        with pytest.raises(ValueError, match="FP32 storage"):
            backend.upload_fp32_linear(bad, dtype=dtype, device=device)
    # Ordinary dense upload stays strict rather than silently widening/casting.
    with pytest.raises(ValueError, match="dtype must match"):
        backend.upload_linear(descriptor, dtype=dtype, device=device)


@pytest.mark.parametrize("tokens", [1, 9])
def test_mixed_fp32_linear_independent_oracle_owned_storage_and_alias(
    mixed_port, tokens
):
    backend, dtype, device = mixed_port
    gen = torch.Generator().manual_seed(8452)
    w = (torch.randn(48, 129, generator=gen) * 0.1).bfloat16().float()
    saved = w.clone()
    r = backend.upload_fp32_linear(DenseWeight(w), dtype=dtype, device=device)
    w.zero_()
    assert torch.equal(r.storage_tensors[0].cpu(), saved)
    x = (torch.randn(tokens, 129, generator=gen) * 0.4).to(device=device, dtype=dtype)
    out = torch.empty(tokens, 48, dtype=dtype, device=device)
    p = r.prepare(tokens=tokens)
    p.run(x, out=out)
    ideal = x.cpu().double() @ saved.double().T
    assert_qualified(out, ideal)
    assert p.workspace_bytes == 0
    # Parameter/input/output intervals remain protected regardless of dtype.
    flat = r.storage_tensors[0].view(dtype).view(-1)
    with pytest.raises(ValueError, match="alias"):
        p.run(x, out=flat[: tokens * 48].view(tokens, 48))


@pytest.mark.parametrize("kernel", [1, 4])
def test_fp32_convolution_preserves_subnormal_bf16_and_chunk_history(
    mixed_port, kernel
):
    backend, dtype, device = mixed_port
    raw = torch.full((kernel, 3), 1e-8, dtype=torch.bfloat16)
    assert not bool(raw.half().any())
    r = backend.upload_causal_conv(raw.float(), dtype=dtype, device=device)
    assert r.dtype == dtype and r._weight.dtype == torch.float32
    assert torch.equal(r._weight.cpu(), raw.float())
    x = torch.full((9, 3), 65504, dtype=dtype, device=device)
    history = torch.zeros(kernel - 1, 3, dtype=dtype, device=device)
    expected, h = conv_oracle(x, raw, history)
    out = torch.empty_like(x)
    next_h = torch.empty_like(history)
    r.prepare(tokens=9).run(x, history=history, out=out, history_out=next_h)
    assert_qualified(out, expected)
    assert bool(out.abs().min() > 0) and torch.equal(next_h.cpu(), h)
    parts = []
    pos = 0
    for t in (2, 1, 6):
        y = torch.empty_like(x[pos : pos + t])
        r.prepare(tokens=t).run(
            x[pos : pos + t], history=history, out=y, history_out=history
        )
        parts.append(y)
        pos += t
    assert torch.equal(torch.cat(parts), out) and torch.equal(history, next_h)


def test_gdn_original_small_parameters_and_parameter_fidelity_tool(
    tmp_path, mixed_port
):
    backend, dtype, device = mixed_port
    cfg = source_config()
    raw = fixture_tensors()
    prefix = "model.language_model.layers.0.linear_attn."
    for role in ("in_proj_a.weight", "in_proj_b.weight", "conv1d.weight"):
        raw[prefix + role].flatten()[0] = torch.tensor(1e-8, dtype=torch.bfloat16)
    source = write_source(tmp_path / "source", raw=cfg, tensors=raw)
    export = export_qwen_q8(source, tmp_path / "q8.ninfer")
    w = QwenQ8Weights.from_ninfer(NInferArtifact(export["output"]))
    r = ResidentQwenGDNLayer.from_q8(w, backend, layer=0, dtype=dtype, device=device)
    assert (
        qwen_resident_bytes(
            w, dtype=dtype, reference=backend.name == "torch-reference", layer=0
        )
        == r.storage_nbytes
    )
    with pytest.raises(ValueError, match="Layer index"):
        qwen_resident_bytes(w, dtype=dtype, layer=True)
    tool = load_tool()
    stats = tool.small_parameter_statistics(w, r)
    assert len(stats) == 3 and all(
        s["changed_elements"] == s["underflow_to_zero"] == 0 for s in stats
    )
    parameter_dtype = torch.float32 if dtype == torch.float16 else dtype
    assert all(s["parameter_dtype"] == str(parameter_dtype) for s in stats)
    for op in (r.a, r.b):
        assert op.dtype == dtype and op.storage_tensors[0].dtype == parameter_dtype
    assert (
        r.frontend.conv.dtype == dtype
        and r.frontend.conv._weight.dtype == parameter_dtype
    )
    # The independent upload checker must reject losing even one original value.
    original = r.a.storage_tensors[0][0, 0].clone()
    r.a.storage_tensors[0][0, 0] = 0
    with pytest.raises(AssertionError, match="small_parameter_fidelity"):
        tool.small_parameter_statistics(w, r)
    r.a.storage_tensors[0][0, 0] = original


def test_fp32_parameter_graph_has_no_fallback_allocations_or_weight_cast(
    mixed_port, monkeypatch
):
    backend, dtype, device = mixed_port
    if device == "cpu":
        pytest.skip("native CUDA capture")
    weight = torch.full((3, 5), 1e-8, dtype=torch.bfloat16).float()
    linear = backend.upload_fp32_linear(
        DenseWeight(weight), dtype=dtype, device=device
    ).prepare(tokens=9)
    conv = backend.upload_causal_conv(weight, dtype=dtype, device=device).prepare(
        tokens=9
    )
    x = torch.full((9, 5), 1024, dtype=dtype, device=device)
    out = torch.empty(9, 3, dtype=dtype, device=device)
    y = torch.empty_like(x)
    history = torch.zeros(2, 5, dtype=dtype, device=device)
    history_out = torch.empty_like(history)

    def run():
        linear.run(x, out=out)
        conv.run(x, history=history, out=y, history_out=history_out)

    run()
    torch.cuda.synchronize()
    expected = tuple(t.clone() for t in (out, y, history_out))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*a, **k):
        raise AssertionError("native FP32 parameter path allocated/cast/fell back")

    with torch.cuda.stream(stream), monkeypatch.context() as m:
        for n in ("empty", "empty_like", "matmul", "mm", "cat"):
            m.setattr(torch, n, forbidden)
        for n in ("to", "float", "copy_", "clone", "reshape", "view"):
            m.setattr(torch.Tensor, n, forbidden)
        with torch.cuda.graph(graph, stream=stream):
            run()
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert all(torch.equal(a, b) for a, b in zip((out, y, history_out), expected))
