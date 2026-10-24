"""Complete native layer composition, state chunking and preflight ownership."""

from dataclasses import replace
import os

import pytest
import torch

from qwen_gdn_reference import gdn_layer_oracle, qualify_layer_operations
from test_qwen_source import fixture_tensors, source_config, write_source
from venturi.models import QwenQ8Weights, ResidentQwenGDNLayer
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, TorchBackend
from venturi.qwen_export import export_qwen_q8


def fixture_weights(tmp_path, hv=4):
    raw, cfg = fixture_tensors(), source_config()
    cfg["text_config"]["linear_num_value_heads"] = hv
    prefix = "model.language_model.layers.0."
    channels, width = 256 + hv * 13, hv * 13
    shapes = {
        "linear_attn.in_proj_qkv.weight": (channels, 32),
        "linear_attn.in_proj_z.weight": (width, 32),
        "linear_attn.in_proj_a.weight": (hv, 32),
        "linear_attn.in_proj_b.weight": (hv, 32),
        "linear_attn.out_proj.weight": (32, width),
        "linear_attn.conv1d.weight": (channels, 1, 4),
        "linear_attn.A_log": (hv,),
        "linear_attn.dt_bias": (hv,),
    }
    for name, shape in shapes.items():
        raw[prefix + name] = torch.empty(shape)
    gen = torch.Generator().manual_seed(103)
    for key, value in list(raw.items()):
        if not key.startswith(prefix):
            continue
        scale = 0.15 if value.ndim == 2 else 0.2 if value.ndim == 3 else 0.05
        tensor = torch.randn(value.shape, generator=gen, dtype=torch.float64) * scale
        if key.endswith("linear_attn.norm.weight"):
            tensor.add_(1)
        if key.endswith("A_log"):
            tensor = torch.linspace(-1.3, 0.9, hv).double()
        if key.endswith("dt_bias"):
            tensor = torch.linspace(-0.8, 1.1, hv).double()
        raw[key] = tensor.to(torch.float32 if key.endswith("A_log") else torch.bfloat16)
    source = write_source(tmp_path / "source", raw=cfg, tensors=raw)
    result = export_qwen_q8(source, tmp_path / "test-q8.ninfer", rows_per_chunk=7)
    return (
        QwenQ8Weights.from_ninfer(NInferArtifact(result["output"])),
        raw,
        cfg["text_config"],
    )


@pytest.fixture(params=["fp64", "cpu-bf16", "cuda"])
def engine(request):
    if request.param != "cuda":
        return (
            TorchBackend(),
            torch.float64 if request.param == "fp64" else torch.bfloat16,
            "cpu",
        )
    path = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not path:
        pytest.skip("explicit native blocks library required")
    return CudaBackend(path), torch.bfloat16, "cuda:0"


def inputs(resident, tokens):
    gen = torch.Generator().manual_seed(81)
    x = (torch.randn(tokens, 32, generator=gen, dtype=torch.float64) * 0.4).to(
        dtype=resident.dtype, device=resident.device
    )
    state = resident.zero_state()
    state.history.copy_(
        torch.randn(state.history.shape, generator=gen).to(resident.device) * 0.05
    )
    state.recurrent.copy_(
        torch.randn(state.recurrent.shape, generator=gen).to(resident.device) * 0.03
    )
    return x, state


def qualified(actual, expected, dtype, floor=2e-5):
    if dtype == torch.float64:
        torch.testing.assert_close(
            actual.cpu().double(), expected.double(), atol=2e-12, rtol=2e-11
        )
    else:
        error = (actual.cpu().double() - expected.to(dtype).double()).abs()
        assert bool(
            (error <= 2 * torch.finfo(dtype).eps * expected.abs() + floor).all()
        ), (error.max().item(), expected.abs().max().item())


@pytest.mark.parametrize("hv", [4, 6])
@pytest.mark.parametrize("tokens", [1, 9])
def test_complete_layer_independent_oracle_and_chunk_state(
    tmp_path, engine, hv, tokens
):
    backend, dtype, device = engine
    weights, raw, cfg = fixture_weights(tmp_path, hv)
    resident = ResidentQwenGDNLayer.from_q8(
        weights, backend, layer=0, dtype=dtype, device=device
    )
    pointers = [t.data_ptr() for t in resident.storage_tensors]
    assert resident.storage_nbytes == sum(
        t.numel() * t.element_size() for t in resident.storage_tensors
    )
    x, state = inputs(resident, tokens)
    plan, output = resident.prepare(tokens=tokens), torch.empty_like(x)
    workspace = plan.empty_workspace()
    assert workspace.nbytes == sum(
        t.numel() * t.element_size() for t in workspace.tensors
    )
    final_history, final_state = (
        torch.empty_like(state.history),
        torch.empty_like(state.recurrent),
    )
    expected, expected_history, expected_state, trace = gdn_layer_oracle(
        raw, cfg, x.cpu(), state.history.cpu(), state.recurrent.cpu()
    )
    plan.run(
        x,
        out=output,
        history=state.history,
        state=state.recurrent,
        history_out=final_history,
        state_out=final_state,
        workspace=workspace,
    )
    if dtype == torch.float64:
        qualified(output, expected, dtype)
        qualified(final_history, expected_history, dtype)
        torch.testing.assert_close(
            final_state.cpu().double(), expected_state, atol=2e-12, rtol=2e-11
        )
    # BF16 whole-chain FP64 error is deliberately diagnostic: a correctly
    # rounded early boundary can change later cancellation/rounding. Qualify
    # every Op against FP64 on its actual inputs instead, without weaker bounds.
    manual, manual_history, manual_state, checks = qualify_layer_operations(
        plan,
        raw,
        x,
        state.history,
        state.recurrent,
        workspace=workspace,
    )
    assert all(c["passed"] for c in checks)
    assert torch.equal(output, manual)
    assert torch.equal(final_history, manual_history)
    assert torch.equal(final_state, manual_state)
    assert not torch.equal(output, x) and not torch.equal(workspace.residual, x)
    chunks, position, parts, keeps = state, 0, [], []
    for count in (1,) if tokens == 1 else (1, 2, 6):
        p = resident.prepare(tokens=count)
        w, out = p.empty_workspace(), torch.empty_like(x[position : position + count])
        reference, ref_history, ref_state, _ = qualify_layer_operations(
            p,
            raw,
            x[position : position + count],
            chunks.history,
            chunks.recurrent,
            workspace=w,
        )
        p.run(
            x[position : position + count],
            out=out,
            history=chunks.history,
            state=chunks.recurrent,
            history_out=chunks.history,
            state_out=chunks.recurrent,
            workspace=w,
        )
        assert torch.equal(out, reference)
        assert torch.equal(chunks.history, ref_history)
        assert torch.equal(chunks.recurrent, ref_state)
        parts.append(out)
        keeps.append((p, w))
        position += count
    # Native dot/recurrent kernels use token-independent reduction order. The
    # explicit CPU BF16 Torch reference uses shape-dependent BLAS algorithms;
    # its chunk outputs are separately qualified above, not claimed bit-exact.
    if dtype == torch.float64 or resident.device.type == "cuda":
        tolerance = 2e-12 if dtype == torch.float64 else 0
        torch.testing.assert_close(
            torch.cat(parts), output, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            chunks.recurrent, final_state, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            chunks.history, final_history, atol=tolerance, rtol=tolerance
        )
    assert pointers == [t.data_ptr() for t in resident.storage_tensors]


def test_all_layer_buffers_preflight_before_any_projection_or_state_write(
    tmp_path, engine
):
    backend, dtype, device = engine
    weights, _, _ = fixture_weights(tmp_path)
    resident = ResidentQwenGDNLayer.from_q8(
        weights, backend, layer=0, dtype=dtype, device=device
    )
    x, state = inputs(resident, 3)
    plan, out = resident.prepare(tokens=3), torch.full_like(x, 19)
    w = plan.empty_workspace()
    for t in w.tensors:
        t.fill_(17)
    originals = tuple(
        t.clone() for t in (x, state.history, state.recurrent, out, *w.tensors)
    )
    cases = (
        replace(w, normalized=x),
        replace(w, projected=out),
        replace(w, z_heads=w.z_heads.clone()),
        replace(w, mlp=replace(w.mlp, gate=w.mlp.up)),
        replace(w, mlp=replace(w.mlp, hidden=w.residual)),
        replace(w, mixer=replace(w.mixer, beta=w.mixer.log_decay)),
        replace(w, projection_scratch=(None,)),
        replace(w, a=w.a.to(torch.int32)),
    )
    for bad in cases:
        with pytest.raises((ValueError, TypeError)):
            plan.run(
                x,
                out=out,
                history=state.history,
                state=state.recurrent,
                history_out=state.history,
                state_out=state.recurrent,
                workspace=bad,
            )
        assert all(
            torch.equal(value, ideal)
            for value, ideal in zip(
                (x, state.history, state.recurrent, out, *w.tensors), originals
            )
        )
    for bad_kwargs in (dict(state_out=state.recurrent.to(torch.int32)), dict(out=x)):
        args = dict(
            out=out,
            history=state.history,
            state=state.recurrent,
            history_out=state.history,
            state_out=state.recurrent,
            workspace=w,
        )
        args.update(bad_kwargs)
        with pytest.raises(ValueError):
            plan.run(x, **args)
    assert all(
        torch.equal(value, ideal)
        for value, ideal in zip(
            (x, state.history, state.recurrent, out, *w.tensors), originals
        )
    )
    # Close rejects even reference-only execution before touching any operand.
    other = resident.prepare(tokens=1)
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.run(
            x,
            out=out,
            history=state.history,
            state=state.recurrent,
            history_out=state.history,
            state_out=state.recurrent,
            workspace=w,
        )
    x1, s1 = inputs(resident, 1)
    other.run(
        x1,
        out=torch.empty_like(x1),
        history=s1.history,
        state=s1.recurrent,
        history_out=s1.history,
        state_out=s1.recurrent,
        workspace=other.empty_workspace(),
    )
    assert all(
        torch.equal(value, ideal)
        for value, ideal in zip(
            (x, state.history, state.recurrent, out, *w.tensors), originals
        )
    )


def test_invalid_layer_or_precision_rejected_before_payload_reads(
    tmp_path, monkeypatch
):
    weights, _, _ = fixture_weights(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid layer started reading weights")

    monkeypatch.setattr(weights.artifact, "read_object", forbidden)
    for layer in (-1, True, 1, 2):
        with pytest.raises(ValueError):
            ResidentQwenGDNLayer.from_q8(
                weights, TorchBackend(), layer=layer, dtype=torch.float64, device="cpu"
            )
    with pytest.raises(ValueError):
        ResidentQwenGDNLayer.from_q8(
            weights, TorchBackend(), layer=0, dtype=torch.float16, device="cpu"
        )


@pytest.mark.cuda
def test_native_layer_graph_has_no_tensor_fallback_or_runtime_views(
    tmp_path, monkeypatch
):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    weights, _, _ = fixture_weights(tmp_path)
    resident = ResidentQwenGDNLayer.from_q8(weights, CudaBackend(library), layer=0)
    plan = resident.prepare(tokens=9)
    x, state = inputs(resident, 9)
    w, out = plan.empty_workspace(), torch.empty_like(x)
    h, s = torch.empty_like(state.history), torch.empty_like(state.recurrent)

    def execute():
        plan.run(
            x,
            out=out,
            history=state.history,
            state=state.recurrent,
            history_out=h,
            state_out=s,
            workspace=w,
        )

    execute()
    expected = tuple(t.clone() for t in (out, h, s))
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Native layer allocated, created a view, or used numerical Torch fallback"
        )

    with torch.cuda.stream(stream):
        with monkeypatch.context() as m:
            for name in (
                "empty",
                "empty_like",
                "zeros",
                "cat",
                "matmul",
                "mm",
                "sigmoid",
            ):
                m.setattr(torch, name, forbidden)
            for name in (
                "to",
                "clone",
                "copy_",
                "contiguous",
                "view",
                "reshape",
                "transpose",
                "exp",
                "sigmoid",
                "__matmul__",
            ):
                m.setattr(torch.Tensor, name, forbidden)
            with torch.cuda.graph(graph, stream=stream):
                execute()
            graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert all(
        torch.equal(actual, ideal) for actual, ideal in zip((out, h, s), expected)
    )


@pytest.mark.cuda
def test_native_layer_independent_streams_and_closed_projection_preflight(tmp_path):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    weights, _, _ = fixture_weights(tmp_path, hv=6)
    resident = ResidentQwenGDNLayer.from_q8(weights, CudaBackend(library), layer=0)
    plan = resident.prepare(tokens=3)
    executions, expected = [], []
    for index in range(2):
        x, s = inputs(resident, 3)
        x.add_(index * 0.3)
        s.recurrent.mul_(index + 1)
        h, r, out = s.history.clone(), s.recurrent.clone(), torch.empty_like(x)
        w = plan.empty_workspace()
        plan.run(
            x, out=out, history=h, state=r, history_out=h, state_out=r, workspace=w
        )
        expected.append((out, h, r))
        executions.append((x, s, torch.empty_like(x), plan.empty_workspace(), w))
    streams = [torch.cuda.Stream() for _ in executions]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for stream, (x, s, out, w, _) in zip(streams, executions):
        with torch.cuda.stream(stream):
            plan.run(
                x,
                out=out,
                history=s.history,
                state=s.recurrent,
                history_out=s.history,
                state_out=s.recurrent,
                workspace=w,
            )
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for (_, s, out, _, _), ideal in zip(executions, expected):
        assert all(
            torch.equal(actual, target)
            for actual, target in zip((out, s.history, s.recurrent), ideal)
        )
    assert not torch.equal(executions[0][2], executions[1][2])
    torch.cuda.synchronize()
    plan.a.close()
    x, s, out, w, _ = executions[0]
    before = tuple(t.clone() for t in (out, s.history, s.recurrent, w.normalized))
    with pytest.raises(RuntimeError, match="closed"):
        plan.run(
            x,
            out=out,
            history=s.history,
            state=s.recurrent,
            history_out=s.history,
            state_out=s.recurrent,
            workspace=w,
        )
    assert all(
        torch.equal(actual, target)
        for actual, target in zip((out, s.history, s.recurrent, w.normalized), before)
    )
