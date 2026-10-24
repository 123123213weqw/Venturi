"""Independent native full-model plumbing; generated weights, not model-quality evidence."""

import os

import pytest
import torch

from decoder_checks import qualified_decoder_composition
from test_llama import llama_fixture
from venturi.loader import load_checkpoint
from venturi.models import LlamaWeights
from venturi.ops import CudaBackend, TorchBackend

pytestmark = pytest.mark.cuda


@pytest.fixture
def backend():
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    return CudaBackend(library)


@pytest.mark.parametrize("tied", [False, True])
def test_native_checkpoint_embedding_decoder_norm_head_chain(tmp_path, backend, tied):
    _, _ = llama_fixture(tmp_path, dtype=torch.bfloat16, tied=tied, sharded=True)
    bound = LlamaWeights.bind(
        load_checkpoint(tmp_path, device="cpu", loader="safetensors")
    )
    model = bound.upload(backend)
    assert (model.head is model.embedding) is tied
    expected_bytes = sum(
        t.numel() * (4 if t.ndim == 1 else 2)
        for t in {
            t.data_ptr(): t
            for t in (
                bound.embedding.data,
                bound.head.data,
                bound.norm,
                *(
                    t
                    for w in bound.layers
                    for t in (
                        w.attention_norm,
                        w.query.data,
                        w.key.data,
                        w.value.data,
                        w.output.data,
                        w.mlp.norm,
                        w.mlp.gate.data,
                        w.mlp.up.data,
                        w.mlp.down.data,
                    )
                ),
            )
        }.values()
    )
    assert model.storage_nbytes == expected_bytes
    session = model.create_session(32, max_tokens=7)
    # Build an independent native composition and qualify all intermediate
    # operations on their actual inputs against FP64, not a cumulative budget.
    ids = [1, 7, 5, 6, 3, 8, 9]
    x = bound.embedding.data[ids].cuda()
    cos, sin = session._cos[:7], session._sin[:7]
    for weights, plan in zip(bound.layers, session._prepare(7)):
        x, _, checks = qualified_decoder_composition(
            plan.at(position=0), weights, x, cos, sin
        )
        assert all(c["passed"] for c in checks.values())
    normalized = torch.empty_like(x)
    model.norm.prepare(tokens=7).run(x, out=normalized)
    ideal = x.cpu().double()
    ideal = (
        ideal
        * torch.rsqrt(ideal.square().mean(-1, keepdim=True) + bound.config.rms_norm_eps)
        * bound.norm.double()
    )
    assert bool(
        (
            (normalized.cpu().double() - ideal.to(torch.bfloat16).double()).abs()
            <= 2 * torch.finfo(torch.bfloat16).eps * ideal.abs() + 2e-5
        ).all()
    )
    head = model.head.prepare(tokens=7)
    expected = torch.empty(7, 37, dtype=torch.bfloat16, device="cuda")
    head.run(normalized, out=expected, workspace=session.workspace.linear)
    ideal_logits = normalized.cpu().double() @ bound.head.data.double().T
    assert bool(
        (
            (expected.cpu().double() - ideal_logits.to(torch.bfloat16).double()).abs()
            <= 2 * torch.finfo(torch.bfloat16).eps * ideal_logits.abs() + 2e-5
        ).all()
    )
    actual = session.forward(ids, all_logits=True)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected.float())
    previous = actual.clone()
    session.forward([2])
    assert torch.equal(previous, actual)  # Returned logits are not scratch views.
    session.reset()
    chunked = [
        session.forward(ids[:3], all_logits=True),
        session.forward(ids[3:], all_logits=True),
    ]
    # Different GEMM/attention reduction shapes need not be bit-identical.
    torch.testing.assert_close(torch.cat(chunked), actual, atol=0.04, rtol=0.04)
    session.close()
    head.close()


def test_native_session_stream_guard_and_lanes(tmp_path, backend):
    llama_fixture(tmp_path, dtype=torch.bfloat16)
    bound = LlamaWeights.bind(
        load_checkpoint(tmp_path, device="cpu", loader="safetensors")
    )
    model = bound.upload(backend)
    a = model.create_session(8, max_tokens=3)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        b = model.create_session(8, max_tokens=3)
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            a.forward([1])
        b_logits = b.forward([1, 2, 3])
        b.close()
    a_logits = a.forward([1, 2, 3])
    assert torch.equal(a_logits, b_logits)
    assert all(
        x._keys.data_ptr() != y._keys.data_ptr() for x, y in zip(a.caches, b.caches)
    )
    a.close()


def test_native_and_reference_fixture_greedy_generation(tmp_path, backend):
    from venturi.generation import generate_tokens

    llama_fixture(tmp_path, dtype=torch.bfloat16)
    bound = LlamaWeights.bind(
        load_checkpoint(tmp_path, device="cpu", loader="safetensors")
    )
    native, reference = (
        bound.upload(backend),
        bound.upload(TorchBackend(), device="cuda:0"),
    )
    a = generate_tokens(native, [1, 5, 9], max_new_tokens=8)
    b = generate_tokens(reference, [1, 5, 9], max_new_tokens=8)
    assert a.generated_ids == b.generated_ids


def test_transformers_cpu_fp64_full_sequence(tmp_path):
    transformers = pytest.importorskip("transformers")
    from test_llama import llama_oracle

    config, tensors = llama_fixture(tmp_path)
    hf_config = transformers.LlamaConfig(**config)
    hf_config._attn_implementation = "eager"
    hf = transformers.LlamaForCausalLM(hf_config).double().eval()
    hf.load_state_dict(tensors, strict=True)
    with torch.inference_mode():
        logits = hf(torch.tensor([[1, 5, 9, 7]])).logits[0]
    # Transformers intentionally softmaxes in FP32 even with FP64 weights;
    # compare that independent implementation within its FP32 arithmetic error.
    torch.testing.assert_close(
        logits, llama_oracle(config, tensors, [1, 5, 9, 7]), atol=2e-6, rtol=2e-6
    )
