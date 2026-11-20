"""Projected-input GDN mixer: independent staged math and per-sequence state."""

from dataclasses import replace
import os

import pytest
import torch

from test_gdn import gdn_oracle
from test_gdn_frontend import frontend_artifact
from test_state_space import conv_oracle, sample, silu64
from venturi.models import GDNFrontendWeights
from venturi.ops import CudaBackend, TorchBackend


@pytest.fixture(params=["reference", "cuda"])
def mixer_backend(request):
    if request.param == "reference":
        return TorchBackend(), torch.float64, "cpu"
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    return CudaBackend(library), torch.bfloat16, "cuda:0"


def mixer_inputs(tokens, dtype, device):
    work = torch.float64 if dtype == torch.float64 else torch.float32
    return tuple(
        sample(shape, dtype if i < 5 else work, device, 100 + i)
        for i, shape in enumerate(
            (
                (tokens, 308),
                (tokens, 4),
                (tokens, 4),
                (tokens, 4, 13),
                (3, 308),
                (4, 13, 64),
            )
        )
    )


def test_staged_mixer_oracle_chunks_and_shared_constants(tmp_path, mixer_backend):
    backend, dtype, device = mixer_backend
    artifact, _ = frontend_artifact(tmp_path)
    weights = GDNFrontendWeights.from_ninfer(artifact, layer=0)
    resident = weights.upload(backend, dtype=dtype, device=device)
    pointers = [t.data_ptr() for t in resident.storage_tensors]
    plan = resident.prepare_mixer_core(tokens=9)
    qkv, a, b, z, history, state = mixer_inputs(9, dtype, device)
    workspace = plan.empty_workspace()
    assert workspace.nbytes == sum(
        t.numel() * t.element_size() for t in workspace.tensors
    )
    out, final_history, final_state = (
        torch.empty_like(z),
        torch.empty_like(history),
        torch.empty_like(state),
    )
    plan.run(
        qkv,
        a,
        b,
        z,
        history=history,
        state=state,
        out=out,
        history_out=final_history,
        state_out=final_state,
        workspace=workspace,
    )
    conv, expected_history = conv_oracle(qkv, weights.conv.to(dtype), history)
    conv = conv.to(dtype)
    q, k, v = (
        conv[:, :128].reshape(9, 2, 64),
        conv[:, 128:256].reshape(9, 2, 64),
        conv[:, 256:].reshape(9, 4, 13),
    )
    work = torch.float64 if dtype == torch.float64 else torch.float32
    g = (
        -weights.a_log.double().exp()
        * torch.log1p((a.cpu().double() + weights.dt_bias.double()).exp())
    ).to(work)
    beta = b.cpu().double().sigmoid().to(dtype).to(work)
    ideal, expected_state = gdn_oracle(q, k, v, g, beta, state)
    value = ideal.to(dtype).double()
    normalized = (
        (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6))
        .to(dtype)
        .double()
    )
    weighted = (normalized * weights.gated_norm.double()).to(dtype).double()
    expected = weighted * silu64(z.cpu().double())
    assert torch.equal(final_history.cpu(), expected_history)
    if dtype == torch.float64:
        torch.testing.assert_close(out, expected, atol=2e-13, rtol=2e-12)
        torch.testing.assert_close(final_state, expected_state, atol=1e-13, rtol=1e-12)
    else:
        error = (out.cpu().double() - expected.to(dtype).double()).abs()
        assert bool((error <= 2 * torch.finfo(dtype).eps * expected.abs() + 2e-5).all())
        torch.testing.assert_close(
            final_state.cpu().double(), expected_state, atol=2e-6, rtol=2e-4
        )
    parts, position = [], 0
    chunk_history, chunk_state = history.clone(), state.clone()
    for count in (1, 2, 6):
        chunk = resident.prepare_mixer_core(tokens=count)
        target = torch.empty_like(z[position : position + count])
        chunk.run(
            *(t[position : position + count] for t in (qkv, a, b, z)),
            history=chunk_history,
            state=chunk_state,
            out=target,
            history_out=chunk_history,
            state_out=chunk_state,
            workspace=chunk.empty_workspace(),
        )
        parts.append(target)
        position += count
    tolerance = 2e-13 if dtype == torch.float64 else 0
    torch.testing.assert_close(torch.cat(parts), out, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(chunk_state, final_state, atol=tolerance, rtol=tolerance)
    assert torch.equal(chunk_history, final_history)
    assert pointers == [t.data_ptr() for t in resident.storage_tensors]


def test_mixer_preflights_every_buffer_before_state_mutation(tmp_path, mixer_backend):
    backend, dtype, device = mixer_backend
    artifact, _ = frontend_artifact(tmp_path)
    resident = GDNFrontendWeights.from_ninfer(artifact, layer=0).upload(
        backend, dtype=dtype, device=device
    )
    plan = resident.prepare_mixer_core(tokens=3)
    qkv, a, b, z, history, state = mixer_inputs(3, dtype, device)
    original_history, original_state = history.clone(), state.clone()
    workspace, out = plan.empty_workspace(), torch.full_like(z, 17)
    for invalid in (
        replace(workspace, convolved=qkv),
        replace(workspace, beta=workspace.log_decay),
        replace(workspace, recurrent=out),
        replace(workspace, beta=workspace.beta.to(torch.int32)),
    ):
        with pytest.raises(ValueError):
            plan.run(
                qkv,
                a,
                b,
                z,
                history=history,
                state=state,
                out=out,
                history_out=history,
                state_out=state,
                workspace=invalid,
            )
        assert torch.equal(history, original_history) and torch.equal(
            state, original_state
        )
        assert bool((out == 17).all())
    # Even identical Tensor objects in two input roles must not weaken aliases:
    # allowing history/history_out in-place must NOT allow clobbering qkv.
    original_qkv = qkv.clone()
    with pytest.raises(ValueError, match="alias"):
        plan.run(
            qkv,
            a,
            b,
            z,
            history=qkv,
            state=state,
            out=out,
            history_out=qkv,
            state_out=state,
            workspace=workspace,
        )
    assert torch.equal(qkv, original_qkv) and torch.equal(state, original_state)
    with pytest.raises(ValueError, match="alias"):
        plan.run(
            qkv,
            a,
            b,
            z,
            history=history,
            state=state,
            out=z,
            history_out=history,
            state_out=state,
            workspace=workspace,
        )
    assert torch.equal(history, original_history) and torch.equal(state, original_state)


@pytest.mark.cuda
def test_native_composed_mixer_capture_and_no_fallback(tmp_path, monkeypatch):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    artifact, _ = frontend_artifact(tmp_path)
    backend, dtype, device = CudaBackend(library), torch.bfloat16, "cuda:0"
    resident = GDNFrontendWeights.from_ninfer(artifact, layer=0).upload(backend)
    plan = resident.prepare_mixer_core(tokens=9)
    qkv, a, b, z, history, state = mixer_inputs(9, dtype, device)
    workspace, out = plan.empty_workspace(), torch.empty_like(z)
    final_history, final_state = torch.empty_like(history), torch.empty_like(state)

    def execute():
        plan.run(
            qkv,
            a,
            b,
            z,
            history=history,
            state=state,
            out=out,
            history_out=final_history,
            state_out=final_state,
            workspace=workspace,
        )

    execute()
    expected = tuple(t.clone() for t in (out, final_history, final_state))
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "composed native mixer allocated or used numerical Torch fallback"
        )

    with torch.cuda.stream(stream):
        with monkeypatch.context() as m:
            for name in ("empty", "empty_like", "zeros", "matmul", "cat"):
                m.setattr(torch, name, forbidden)
            for name in (
                "to",
                "clone",
                "copy_",
                "contiguous",
                "view",
                "reshape",
                "exp",
                "sigmoid",
            ):
                m.setattr(torch.Tensor, name, forbidden)
            with torch.cuda.graph(graph, stream=stream):
                execute()
            graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert all(
        torch.equal(actual, ideal)
        for actual, ideal in zip((out, final_history, final_state), expected)
    )


def test_mixer_prepare_rejects_inconsistent_residents(tmp_path):
    artifact, _ = frontend_artifact(tmp_path)
    resident = GDNFrontendWeights.from_ninfer(artifact, layer=0).upload(
        TorchBackend(), dtype=torch.float64, device="cpu"
    )
    for bad in (
        replace(resident, qk_heads=1),
        replace(resident, gates=replace(resident.gates, backend=TorchBackend())),
        replace(resident, gates=replace(resident.gates, dtype=torch.bfloat16)),
        replace(resident, gates=replace(resident.gates, _bias=torch.ones(3))),
    ):
        with pytest.raises(ValueError):
            bad.prepare_mixer_core(tokens=3)


@pytest.mark.cuda
def test_mixer_independent_streams_and_request_states(tmp_path):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    artifact, _ = frontend_artifact(tmp_path)
    resident = GDNFrontendWeights.from_ninfer(artifact, layer=0).upload(
        CudaBackend(library)
    )
    plan = resident.prepare_mixer_core(tokens=9)
    executions, expected = [], []
    for index in range(2):
        qkv, a, b, z, history, state = mixer_inputs(9, torch.bfloat16, "cuda:0")
        qkv.mul_(index + 1)
        state.add_(index * 0.05)
        h, s, out = history.clone(), state.clone(), torch.empty_like(z)
        plan.run(
            qkv,
            a,
            b,
            z,
            history=h,
            state=s,
            out=out,
            history_out=h,
            state_out=s,
            workspace=plan.empty_workspace(),
        )
        expected.append((out, h, s))
        executions.append(
            (qkv, a, b, z, history, state, torch.empty_like(z), plan.empty_workspace())
        )
    streams = [torch.cuda.Stream() for _ in executions]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for stream, (qkv, a, b, z, history, state, out, workspace) in zip(
        streams, executions
    ):
        with torch.cuda.stream(stream):
            plan.run(
                qkv,
                a,
                b,
                z,
                history=history,
                state=state,
                out=out,
                history_out=history,
                state_out=state,
                workspace=workspace,
            )
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for (_, _, _, _, history, state, out, _), ideal in zip(executions, expected):
        assert all(
            torch.equal(actual, target)
            for actual, target in zip((out, history, state), ideal)
        )
    assert not torch.equal(executions[0][6], executions[1][6])
