"""Resident storage is owned once and retained independently by shape plans."""

from dataclasses import FrozenInstanceError
import gc
import os

import pytest
import torch

from block_checks import qualified_native_composition
from block_reference import gated_mlp
from test_gated_mlp import make_weights
from venturi.ops import AdaBackend, CudaBackend, DenseWeight, TorchBackend


@pytest.mark.parametrize("q4", [False, True])
def test_reference_resident_shapes_lifetime_and_source_isolation(q4, monkeypatch):
    weights = make_weights(16, 32, q4=q4)
    expected = [
        (x := torch.randn(t, 16, dtype=torch.float64), gated_mlp(weights, x, 1e-6))
        for t in (1, 7, 17)
    ]
    resident = weights.upload(TorchBackend(), dtype=torch.float64, device="cpu")
    pointers = tuple(t.data_ptr() for t in resident.storage_tensors)
    nbytes = resident.storage_nbytes

    # Binding must not copy weights, allocate tensors, or redo Q4 materialization.
    def forbidden(*args, **kwargs):
        raise AssertionError("shape binding attempted a tensor allocation/copy")

    with monkeypatch.context() as m:
        for name in ("to", "clone", "contiguous"):
            m.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "zeros", "ones"):
            m.setattr(torch, name, forbidden)
        plans = [resident.prepare(tokens=t) for t in (1, 7, 17)]
    assert nbytes == sum(t.numel() * t.element_size() for t in plans[0].storage_tensors)
    for plan in plans:
        assert tuple(t.data_ptr() for t in plan.storage_tensors) == pointers
    weights.norm.zero_()
    for weight in (weights.gate, weights.up, weights.down):
        if q4:
            weight.scales.zero_()
            weight.codes.zero_()
        else:
            weight.data.zero_()
    del resident, weights
    gc.collect()
    for plan, (x, ref) in zip(plans, expected):
        out = torch.empty_like(x)
        plan.run(x, out=out, workspace=plan.empty_workspace())
        torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    with pytest.raises(FrozenInstanceError):
        plans[0].tokens = 8


def test_resident_binding_rejects_invalid_shape_without_allocating():
    resident = TorchBackend().upload_linear(
        DenseWeight(torch.ones(3, 5)), dtype=torch.float32, device="cpu"
    )
    for tokens in (0, -1, True, 1.5, 2**31):
        with pytest.raises(ValueError, match="positive int32"):
            resident.prepare(tokens=tokens)


@pytest.mark.cuda
@pytest.mark.parametrize("q4", [False, True])
def test_cuda_resident_shapes_stream_graph_and_handle_independence(q4, monkeypatch):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    ada_library = os.environ.get("VENTURI_ADA_LIBRARY")
    if not library or (q4 and not ada_library):
        pytest.skip("explicit native libraries required")
    backend = CudaBackend(library, q4_backend=AdaBackend(ada_library) if q4 else None)
    weights = make_weights(65, 97, dtype=torch.bfloat16, q4=q4)
    resident = weights.upload(backend, dtype=torch.bfloat16, device="cuda:0")
    pointers = tuple(t.data_ptr() for t in resident.storage_tensors)

    def forbidden(*args, **kwargs):
        raise AssertionError("shape binding attempted a tensor allocation/copy")

    before = torch.cuda.memory_allocated()
    with monkeypatch.context() as m:
        for name in ("to", "clone", "contiguous"):
            m.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "zeros", "ones"):
            m.setattr(torch, name, forbidden)
        plans = [resident.prepare(tokens=t) for t in (1, 7, 17)]
    assert torch.cuda.memory_allocated() == before
    for plan in plans:
        assert tuple(t.data_ptr() for t in plan.storage_tensors) == pointers
    # A handle belongs to a shape, not to shared storage.
    plans[0].close()
    del resident
    gc.collect()
    for plan in plans[1:]:
        x = torch.randn(plan.tokens, 65, dtype=torch.bfloat16, device="cuda")
        expected = qualified_native_composition(plan, weights, x)
        ws, out = plan.empty_workspace(), torch.empty_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan.run(x, out=out, workspace=ws)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan.run(x, out=out, workspace=ws)
            graph.replay()
        stream.synchronize()
        # Every actual Op input passed FP64 qualification; composition is exact.
        assert torch.equal(out, expected)
        del graph
        plan.close()


@pytest.mark.cuda
def test_cuda_resident_binding_capture_is_rejected():
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    resident = CudaBackend(library).upload_linear(
        DenseWeight(torch.ones(3, 5, dtype=torch.float16)),
        dtype=torch.float16,
        device="cuda:0",
    )
    # Exercise the guard without invalidating an actual graph capture.
    from unittest.mock import patch

    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
            resident.prepare(tokens=1)


@pytest.mark.cuda
def test_direct_ada_residency_shares_fp16_bf16_shapes_and_routes(monkeypatch):
    library = os.environ.get("VENTURI_ADA_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    weight = make_weights(65, 97, q4=True).gate
    decoded = weight.dequantize_reference().double()
    resident = AdaBackend(library).upload_linear(weight)

    def forbidden(*args, **kwargs):
        raise AssertionError("binding copied weights")

    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "to", forbidden)
        a = resident.prepare(tokens=1, dtype=torch.float16, route="generic")
        b = resident.prepare(tokens=7, dtype=torch.bfloat16, route="auto")
    assert a._weight is b._weight
    assert resident.storage_nbytes == sum(t.nbytes for t in resident.storage_tensors)
    weight.scales.zero_()
    del resident
    gc.collect()
    for plan in (a, b):
        x = torch.randn(plan.tokens, 65, dtype=plan.dtype, device="cuda")
        out = plan.empty_output()
        expected = (x.cpu().double() @ decoded.T).to(plan.dtype)
        plan.run(x, out=out)
        torch.testing.assert_close(
            out.cpu(), expected, rtol=2 * torch.finfo(plan.dtype).eps, atol=2e-4
        )


@pytest.mark.parametrize("old_abi", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
def test_blocks_checks_abi_before_binding_new_symbols(tmp_path, monkeypatch, old_abi):
    from unittest.mock import Mock

    library = tmp_path / "old.so"
    library.touch()
    abi = Mock(return_value=old_abi)

    class OldLibrary:
        venturi_blocks_abi = abi

    monkeypatch.setattr("venturi.ops.blocks.ctypes.CDLL", lambda _: OldLibrary())
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        CudaBackend(library)


@pytest.mark.cuda
@pytest.mark.parametrize("q4", [False, True])
def test_shared_weights_overlapping_streams_independent_workspaces(q4):
    library, ada = (
        os.environ.get("VENTURI_BLOCKS_LIBRARY"),
        os.environ.get("VENTURI_ADA_LIBRARY"),
    )
    if not library or (q4 and not ada):
        pytest.skip("explicit native libraries required")
    backend = CudaBackend(library, q4_backend=AdaBackend(ada) if q4 else None)
    weights = make_weights(65, 97, dtype=torch.bfloat16, q4=q4)
    resident = weights.upload(backend, dtype=torch.bfloat16, device="cuda:0")
    plan = resident.prepare(tokens=7)
    inputs = [torch.randn(7, 65, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    expected = [qualified_native_composition(plan, weights, x) for x in inputs]
    targets = [torch.empty_like(x) for x in inputs]
    workspaces = [plan.empty_workspace() for _ in inputs]
    streams = [torch.cuda.Stream() for _ in inputs]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(16):
        for stream, x, target, ws in zip(streams, inputs, targets, workspaces):
            with torch.cuda.stream(stream):
                plan.run(x, out=target, workspace=ws)
    for stream in streams:
        stream.synchronize()
    assert all(torch.equal(out, ref) for out, ref in zip(targets, expected))
    plan.close()
