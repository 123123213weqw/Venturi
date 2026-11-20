"""Q8 artifact bindings, non-multiple-of-128 padding and shared resident weights."""

from dataclasses import replace
import os

import pytest
import torch

from venturi.container.safetensors import SafetensorsContainer
from venturi.quantization.layout import row_geometry
from venturi.ops import CudaBackend, Q8VocabularyWeights, TorchBackend


def vocabulary_weights(n=31, k=129):
    gen = torch.Generator().manual_seed(834)
    padded = (k + 127) // 128 * 128
    codes = torch.randint(-127, 128, (n, padded), generator=gen, dtype=torch.int8)
    scales = (torch.rand(n, padded // 32, generator=gen) * 0.004).half()
    return Q8VocabularyWeights(codes, scales, k)


@pytest.fixture(params=["reference", "cuda"])
def backend(request):
    if request.param == "reference":
        return TorchBackend(), "cpu"
    library = os.environ.get("VENTURI_OPS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    return CudaBackend(library), "cuda:0"


@pytest.mark.parametrize("k", [65, 128, 129, 5120])
def test_q8_embedding_head_independent_fp64_and_shared_storage(backend, k):
    engine, device = backend
    weights = vocabulary_weights(k=k)
    resident = weights.upload(engine, device=device)
    pointers = tuple(t.data_ptr() for t in resident.storage_tensors)
    plans = [
        resident.prepare_embedding([0, 30, 7, 7]),
        resident.prepare(tokens=1),
        resident.prepare(tokens=3),
    ]
    assert all(p.resident is resident for p in plans)
    assert resident.storage_nbytes == weights.codes.numel() + weights.scales.numel() * 2
    exact = (
        weights.codes.double() * weights.scales.double().repeat_interleave(32, -1)
    )[:, :k]
    embedding = torch.empty(4, k, dtype=torch.bfloat16, device=device)
    plans[0].run(out=embedding)
    assert torch.equal(embedding.cpu(), exact[[0, 30, 7, 7]].bfloat16())
    for count, plan in zip((1, 3), plans[1:]):
        x = torch.randn(
            count, k, generator=torch.Generator().manual_seed(311), dtype=torch.bfloat16
        ).to(device)
        guard = torch.full((count * 31 + 2,), 47, dtype=torch.bfloat16, device=device)
        out = guard[1:-1].reshape(count, 31)
        plan.run(x, out=out)
        ideal = x.cpu().double() @ exact.T
        error = (out.cpu().double() - ideal.bfloat16().double()).abs()
        assert bool(
            (error <= 2 * torch.finfo(torch.bfloat16).eps * ideal.abs() + 2e-4).all()
        )
        assert guard[0].item() == guard[-1].item() == 47
    assert pointers == tuple(t.data_ptr() for t in resident.storage_tensors)


def test_standard_tensor_plane_round_trip(tmp_path):
    from safetensors.torch import save_file

    weights = vocabulary_weights()
    path = tmp_path / "vocabulary.safetensors"
    save_file({"codes": weights.codes, "scales": weights.scales}, str(path))
    source = SafetensorsContainer(path)
    codes = torch.frombuffer(
        bytearray(source.read_bytes("codes")), dtype=torch.int8
    ).reshape(weights.codes.shape)
    scales = torch.frombuffer(
        bytearray(source.read_bytes("scales")), dtype=torch.float16
    ).reshape(weights.scales.shape)
    actual = Q8VocabularyWeights(codes, scales, weights.hidden_size)
    actual.validate()
    assert torch.equal(actual.codes, weights.codes) and torch.equal(
        actual.scales, weights.scales
    )


@pytest.mark.parametrize(
    "mutation", ["minus128", "negative_scale", "nan_scale", "wrong_scale", "padding"]
)
def test_invalid_planes_rejected(mutation):
    weights = vocabulary_weights()
    if mutation == "minus128":
        weights.codes[0, 0] = -128
    elif mutation == "negative_scale":
        weights.scales[0, 0] = -1
    elif mutation == "nan_scale":
        weights.scales[0, 0] = torch.nan
    elif mutation == "wrong_scale":
        weights = replace(weights, scales=weights.scales[:, :-1].contiguous())
    else:
        weights = replace(weights, hidden_size=65)
    with pytest.raises(ValueError):
        weights.upload(TorchBackend(), device="cpu")


def test_q8_invalid_ids_and_aliases(backend):
    engine, device = backend
    resident = vocabulary_weights(n=128, k=128).upload(engine, device=device)
    for ids in ([], [128], [-1], [True]):
        with pytest.raises(ValueError):
            resident.prepare_embedding(ids)
    x = torch.zeros(1, 128, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="alias"):
        resident.prepare(tokens=1).run(x, out=x)


@pytest.mark.cuda
def test_q8_graph_capture_no_dense_fallback(monkeypatch):
    library = os.environ.get("VENTURI_OPS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    resident = vocabulary_weights().upload(CudaBackend(library))
    lookup, head = resident.prepare_embedding([1, 5, 2]), resident.prepare(tokens=3)
    x, out = (
        torch.empty(3, 129, dtype=torch.bfloat16, device="cuda"),
        torch.empty(3, 31, dtype=torch.bfloat16, device="cuda"),
    )
    lookup.run(out=x)
    head.run(x, out=out)
    expected_x, expected = x.clone(), out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("Q8 native execution allocated/materialized/fell back")

    with torch.cuda.stream(stream), monkeypatch.context() as m:
        for name in ("empty", "empty_like", "mm", "matmul", "cat"):
            m.setattr(torch, name, forbidden)
        for name in ("to", "clone", "copy_", "float", "repeat_interleave"):
            m.setattr(torch.Tensor, name, forbidden)
        with torch.cuda.graph(graph, stream=stream):
            lookup.run(out=x)
            head.run(x, out=out)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(x, expected_x)
    assert torch.equal(out, expected)


def test_explicit_fp16_upload_and_unsupported_target():
    weights = vocabulary_weights()
    resident = weights.upload(TorchBackend(), device="cpu", dtype=torch.float16)
    assert resident.dtype == torch.float16

    class Unsupported:
        def validate_device(self, dtype, device):
            raise ValueError("unsupported architecture/precision")

    with pytest.raises(ValueError, match="unsupported architecture"):
        weights.upload(Unsupported(), dtype=torch.float16, device="cuda:0")
