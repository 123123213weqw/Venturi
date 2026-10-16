"""Opt-in real FP16 generation checks; never download checkpoint files."""

import os

import pytest
import torch

from venturi import load_checkpoint
from venturi.generation import LocalTokenizer, generate_text
from venturi.models import GPTNeoXModel


pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def real_model():
    path = os.environ.get("VENTURI_GENERATION_MODEL")
    if not path:
        pytest.skip(
            "Set VENTURI_GENERATION_MODEL to the local FP16 GPT-NeoX checkpoint"
        )
    if not torch.cuda.is_available():
        pytest.fail("VENTURI_GENERATION_MODEL is set but CUDA is unavailable")
    checkpoint = load_checkpoint(path, device="cuda:0", require_float_dtype="F16")
    return GPTNeoXModel(checkpoint)


def test_real_generation_and_request_isolation(real_model):
    tokenizer = LocalTokenizer(
        real_model._checkpoint.checkpoint.directory / "tokenizer.json",
        vocab_size=real_model.config.vocab_size,
    )
    first = generate_text(real_model, tokenizer, "Once upon a time", max_new_tokens=8)
    second = generate_text(real_model, tokenizer, "Once upon a time", max_new_tokens=8)
    assert first.generated_ids == second.generated_ids
    assert first.text and first.text == second.text
    assert 1 <= len(first.generated_ids) <= 8


def test_real_nondefault_stream_and_cache_ownership(real_model):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = real_model.create_session(8)
        result = a.forward([2, 3, 4])
        assert result.shape == (real_model.config.vocab_size,)
    with pytest.raises(ValueError, match="owning CUDA stream"):
        a.forward([5])
    with pytest.raises(ValueError, match="owning CUDA stream"):
        a.reset()
    stream.synchronize()


def test_real_cached_chunk_matches_full_prefill(real_model):
    ids = [2, 5, 11, 17, 23]
    a, b = real_model.create_session(8), real_model.create_session(8)
    a.forward(ids[:2])
    actual = a.forward(ids[2:], all_logits=True)
    expected = b.forward(ids, all_logits=True)[2:]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.04)
