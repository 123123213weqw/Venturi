"""Independent FP64 matrix recurrence, chunk/state/alias qualification."""

import os

import pytest
import torch

from venturi.ops import CudaBackend, DenseWeight, TorchBackend


def gdn_inputs(tokens=9, hq=2, hv=4, dk=64, dv=13, dtype=torch.float64, device="cpu"):
    gen = torch.Generator().manual_seed(945)

    def rand(shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64)

    gate_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    return tuple(
        x.to(device=device, dtype=dt).contiguous()
        for x, dt in (
            (rand((tokens, hq, dk)) * 0.2, dtype),
            (rand((tokens, hq, dk)) * 0.2, dtype),
            (rand((tokens, hv, dv)), dtype),
            (-rand((tokens, hv)).abs() * 0.2, gate_dtype),
            (rand((tokens, hv)).sigmoid(), gate_dtype),
            (rand((hv, dv, dk)) * 0.02, gate_dtype),
        )
    )


def gdn_oracle(q, k, v, g, beta, state, *, normalize=True):
    # Matrices per head, not the production vectorized reference implementation.
    q, k, v, g, beta, state = (x.cpu().double() for x in (q, k, v, g, beta, state))
    state = state.clone()
    t, hv, dv = v.shape
    dk, hq = k.shape[-1], k.shape[1]
    out = torch.empty(t, hv, dv, dtype=torch.float64)
    for pos in range(t):
        for head in range(hv):
            query, key = q[pos, head // (hv // hq)], k[pos, head // (hv // hq)]
            if normalize:
                query = query / torch.sqrt(query @ query + 1e-6)
                key = key / torch.sqrt(key @ key + 1e-6)
            decayed = g[pos, head].exp() * state[head]
            delta = beta[pos, head] * (v[pos, head] - decayed @ key)
            state[head] = decayed + torch.outer(delta, key)
            out[pos, head] = (state[head] @ query) / dk**0.5
    return out, state


@pytest.fixture(params=["reference", "cuda"])
def gdn_backend(request):
    if request.param == "reference":
        return TorchBackend(), torch.float64, "cpu"
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    return CudaBackend(library), torch.bfloat16, "cuda:0"


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dk", [64, 128, 256])
def test_recurrence_fp64_oracle_and_chunks(gdn_backend, normalize, dk):
    backend, dtype, device = gdn_backend
    q, k, v, g, beta, state = gdn_inputs(dk=dk, dtype=dtype, device=device)
    # Raw long recurrences can grow; use bounded raw keys in this case.
    if not normalize:
        k.mul_(0.1)
    expected, expected_state = gdn_oracle(q, k, v, g, beta, state, normalize=normalize)
    plan = backend.prepare_gdn(
        tokens=9,
        qk_heads=2,
        value_heads=4,
        key_dim=dk,
        value_dim=13,
        normalize_qk=normalize,
        dtype=dtype,
        device=device,
    )
    output, next_state = torch.empty_like(v), torch.empty_like(state)
    plan.run(q, k, v, g, beta, state=state, out=output, state_out=next_state)
    if dtype == torch.float64:
        torch.testing.assert_close(output, expected, atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(next_state, expected_state, atol=1e-14, rtol=1e-12)
    else:
        actual = output.cpu().double()
        rounded = expected.to(dtype).double()
        assert bool(
            (
                (actual - rounded).abs()
                <= 2 * torch.finfo(dtype).eps * expected.abs() + 2e-5
            ).all()
        )
        torch.testing.assert_close(
            next_state.cpu().double(), expected_state, atol=2e-6, rtol=2e-4
        )
    chunk_state, parts, position = state.clone(), [], 0
    for count in (3, 1, 5):
        chunk = backend.prepare_gdn(
            tokens=count,
            qk_heads=2,
            value_heads=4,
            key_dim=dk,
            value_dim=13,
            normalize_qk=normalize,
            dtype=dtype,
            device=device,
        )
        part = torch.empty_like(v[position : position + count])
        chunk.run(
            *(x[position : position + count] for x in (q, k, v, g, beta)),
            state=chunk_state,
            out=part,
            state_out=chunk_state,
        )
        parts.append(part)
        position += count
    torch.testing.assert_close(torch.cat(parts), output, atol=0, rtol=0)
    torch.testing.assert_close(chunk_state, next_state, atol=0, rtol=0)


def test_embedding_shared_head_storage_and_ids(gdn_backend):
    backend, dtype, device = gdn_backend
    source = torch.arange(7 * 64, dtype=dtype).reshape(7, 64)
    resident = backend.upload_linear(DenseWeight(source), dtype=dtype, device=device)
    pointers = [t.data_ptr() for t in resident.storage_tensors]
    plan = resident.prepare_embedding([6, 1, 1, 0])
    assert plan._weight.data_ptr() == pointers[0]
    out = torch.empty(4, 64, dtype=dtype, device=device)
    plan.run(out=out)
    assert torch.equal(out.cpu(), source[[6, 1, 1, 0]])
    for ids in ([], [-1], [7], [True], torch.tensor([1])):
        with pytest.raises(ValueError):
            resident.prepare_embedding(ids)
    with pytest.raises(ValueError, match="alias"):
        plan.run(out=resident.storage_tensors[0][:4])
    assert pointers == [t.data_ptr() for t in resident.storage_tensors]


def test_gdn_alias_and_validation(gdn_backend):
    backend, dtype, device = gdn_backend
    q, k, v, g, beta, state = gdn_inputs(dtype=dtype, device=device)
    plan = backend.prepare_gdn(
        tokens=9,
        qk_heads=2,
        value_heads=4,
        value_dim=13,
        key_dim=64,
        dtype=dtype,
        device=device,
    )
    out = torch.empty_like(v)
    with pytest.raises(ValueError, match="alias"):
        plan.run(q, k, v, g, beta, state=state, out=v, state_out=state)
    with pytest.raises(ValueError, match="dtype"):
        plan.run(
            q, k, v, g, beta, state=state, out=out, state_out=state.to(torch.int32)
        )
    for options in (
        {"qk_heads": 3},
        {"key_dim": 32},
        {"scale": float("nan")},
        {"normalize_qk": 1},
    ):
        cfg = dict(
            tokens=9, qk_heads=2, value_heads=4, key_dim=64, dtype=dtype, device=device
        )
        cfg.update(options)
        with pytest.raises(ValueError):
            backend.prepare_gdn(**cfg)


@pytest.mark.cuda
def test_native_gdn_graph_and_no_numerical_fallback(monkeypatch):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    backend = CudaBackend(library)
    q, k, v, g, beta, state = gdn_inputs(dtype=torch.bfloat16, device="cuda")
    plan = backend.prepare_gdn(
        tokens=9, qk_heads=2, value_heads=4, key_dim=64, value_dim=13
    )
    out, final = torch.empty_like(v), torch.empty_like(state)
    plan.run(q, k, v, g, beta, state=state, out=out, state_out=final)
    expected, expected_state = out.clone(), final.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("native GDN used numerical fallback/allocation")

    with torch.cuda.stream(stream):
        with monkeypatch.context() as m:
            for name in ("empty", "empty_like", "matmul", "mm", "cat"):
                m.setattr(torch, name, forbidden)
            for name in ("to", "clone", "copy_", "exp", "__matmul__"):
                m.setattr(torch.Tensor, name, forbidden)
            with torch.cuda.graph(graph, stream=stream):
                plan.run(q, k, v, g, beta, state=state, out=out, state_out=final)
            graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(out, expected)
    assert torch.equal(final, expected_state)


def test_transformers_raw_recurrence_independent_layout():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_recurrent_gated_delta_rule,
    )

    q, k, v, g, beta, state = gdn_inputs(dtype=torch.bfloat16)
    k.mul_(0.1)
    expected, final = gdn_oracle(q, k, v, g, beta, state, normalize=False)
    # HF stores [key,value], ours stores [value,key]. Head expansion is normally
    # done by the HF layer, not inside its recurrence function. Disable q/k
    # normalization to avoid conflating HF's BF16 normalization boundary with
    # the explicitly FP32-normalizing primitive contract.
    actual, actual_state = torch_recurrent_gated_delta_rule(
        q.repeat_interleave(2, dim=1)[None],
        k.repeat_interleave(2, dim=1)[None],
        v[None],
        g[None],
        beta[None],
        initial_state=state.transpose(-1, -2)[None].contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    error = (actual[0].double() - expected.bfloat16().double()).abs()
    assert bool(
        (error <= 2 * torch.finfo(torch.bfloat16).eps * expected.abs() + 2e-5).all()
    )
    torch.testing.assert_close(
        actual_state[0].transpose(-1, -2).double(), final, atol=2e-6, rtol=2e-4
    )


@pytest.mark.cuda
def test_gdn_zero_update_and_independent_streams():
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    backend = CudaBackend(library)
    q, k, v, g, beta, state = gdn_inputs(dtype=torch.bfloat16, device="cuda")
    g.zero_()
    beta.zero_()
    plan = backend.prepare_gdn(
        tokens=9, qk_heads=2, value_heads=4, key_dim=64, value_dim=13
    )
    outputs = [(torch.empty_like(v), torch.empty_like(state)) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for stream, (out, final) in zip(streams, outputs):
        with torch.cuda.stream(stream):
            plan.run(q, k, v, g, beta, state=state, out=out, state_out=final)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert all(torch.equal(final, state) for _, final in outputs)
    assert torch.equal(outputs[0][0], outputs[1][0])
    expected, _ = gdn_oracle(q, k, v, g, beta, state)
    assert bool(
        (
            (outputs[0][0].cpu().double() - expected.bfloat16().double()).abs()
            <= 2 * torch.finfo(torch.bfloat16).eps * expected.abs() + 2e-5
        ).all()
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_dense_embedding_scalar_dispatch(dtype):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    backend = CudaBackend(library)
    weight = torch.randn(17, 65, dtype=dtype)
    resident = backend.upload_linear(DenseWeight(weight), dtype=dtype, device="cuda:0")
    plan = resident.prepare_embedding([16, 0, 5])
    out = torch.empty(3, 65, dtype=dtype, device="cuda")
    plan.run(out=out)
    assert torch.equal(out.cpu(), weight[[16, 0, 5]])
    # Defensive kernel bound guard even when a caller violates immutable IDs.
    plan._indices.fill_(-1)
    plan.run(out=out)
    assert torch.count_nonzero(out).item() == 0
