"""Q8 artifact bindings, non-multiple-of-128 padding and shared resident weights."""

from dataclasses import replace
import os

import pytest
import torch

from test_ninfer import write_artifact
from venturi.ninfer import NInferArtifact, row_geometry
from venturi.ops import CudaBackend, NInferVocabulary, Q8VocabularyWeights, TorchBackend


def vocabulary_weights(n=31, k=129):
    gen = torch.Generator().manual_seed(834)
    padded = (k + 127) // 128 * 128
    codes = torch.randint(-127, 128, (n, padded), generator=gen, dtype=torch.int8)
    scales = (torch.rand(n, padded // 32, generator=gen) * 0.004).half()
    return Q8VocabularyWeights(codes, scales, k)


def vocabulary_artifact(path, weights, *, split=None):
    n, padded = weights.codes.shape
    _, _, _, _, scale_offset, size = row_geometry(
        (n, weights.hidden_size), "q8_g32_fp16"
    )
    code_bytes, scale_bytes = (
        weights.codes.numpy().tobytes(),
        weights.scales.numpy().tobytes(),
    )
    payload = code_bytes + bytes(scale_offset - len(code_bytes)) + scale_bytes
    assert len(payload) == size
    manifest = {
        "components": {
            "text": {
                "config": {
                    "model_type": "qwen3_5_text",
                    "vocab_size": n,
                    "hidden_size": weights.hidden_size,
                    "tie_word_embeddings": True,
                }
            }
        },
        "objects": [
            {
                "id": "vocabulary",
                "kind": "tensor",
                "shape": [n, weights.hidden_size],
                "format": "q8_g32_fp16",
                "layout": "row_split_k128_v1",
                "offset": 0,
                "bytes": size,
            }
        ],
        "bindings": {
            "text/token_embedding": {"object": "vocabulary"},
            "text/output_head": {"object": "vocabulary"},
        },
        "uses": [],
    }
    return NInferArtifact(
        write_artifact(path, manifest=manifest, payload=payload, split=split)
    )


@pytest.fixture(params=["reference", "cuda"])
def backend(request):
    if request.param == "reference":
        return TorchBackend(), "cpu"
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
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


@pytest.mark.parametrize("split", [None, 3, 3980])
def test_ninfer_checkpoint_binding_planes_and_row_decode(tmp_path, split):
    weights = vocabulary_weights()
    artifact = vocabulary_artifact(tmp_path, weights, split=split)
    actual = Q8VocabularyWeights.from_ninfer(artifact, "text/token_embedding")
    assert torch.equal(actual.codes, weights.codes)
    assert torch.equal(actual.scales, weights.scales)
    assert actual.hidden_size == weights.hidden_size
    for row in (0, 30):
        exact = weights.codes[row].float() * weights.scales[
            row
        ].float().repeat_interleave(32)
        assert torch.equal(
            torch.tensor(artifact.decode_row("vocabulary", row)), exact[:129]
        )
    artifact.manifest["bindings"]["text/token_embedding"] = {
        "parts": [{"object": "vocabulary", "range": [0, 129]}]
    }
    with pytest.raises(ValueError, match="direct"):
        Q8VocabularyWeights.from_ninfer(artifact, "text/token_embedding")


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
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
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


def test_vocabulary_binding_deduplicates_and_prevalidates(tmp_path, monkeypatch):
    weights = vocabulary_weights()
    artifact = vocabulary_artifact(tmp_path, weights)
    bound = NInferVocabulary.upload_from_ninfer(artifact, TorchBackend(), device="cpu")
    assert bound.embedding is bound.head
    assert bound.storage_nbytes == bound.embedding.storage_nbytes
    artifact.manifest["components"]["text"]["config"]["hidden_size"] = 128

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid metadata triggered a payload read")

    monkeypatch.setattr(artifact, "read_object", forbidden)
    with pytest.raises(ValueError, match="row-split"):
        NInferVocabulary.upload_from_ninfer(artifact, TorchBackend(), device="cpu")
