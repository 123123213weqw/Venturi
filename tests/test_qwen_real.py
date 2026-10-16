"""Opt-in existing-checkpoint/backend checks; never download weights."""

import os

import pytest

from venturi.backends.qwen import QwenNativeSession
from venturi.ninfer import NInferArtifact


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("VENTURI_QWEN_MODEL")
    if not path:
        pytest.skip("Set VENTURI_QWEN_MODEL to the existing NInfer checkpoint")
    return path


def test_real_quantized_artifact_and_embedded_resources(checkpoint):
    artifact = NInferArtifact(checkpoint)
    text = artifact.manifest["components"]["text"]
    config = text["config"]
    assert config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert len(config["layer_types"]) == config["num_hidden_layers"]
    assert set(config["layer_types"]) <= {"linear_attention", "full_attention"}
    tokenizer = artifact.read_object(text["resources"]["tokenizer.json"])
    assert b'"model"' in tokenizer
    assert artifact.summary()["formats"]["q4_g64_fp16"] > 0


@pytest.mark.cuda
def test_real_native_qwen_graph_session_reuse(checkpoint):
    runner = os.environ.get("VENTURI_QWEN_RUNNER")
    if os.environ.get("VENTURI_TEST_CUDA") != "1" or not runner:
        pytest.skip("Set VENTURI_TEST_CUDA=1 and VENTURI_QWEN_RUNNER")
    with QwenNativeSession(
        checkpoint,
        runner=runner,
        max_context=512,
        prefill_chunk=128,
        cuda_graph=True,
        timeout=600,
    ) as model:
        prompt = "Explain why the sky looks blue. Give a detailed answer."
        first = model.generate(prompt, max_new_tokens=16)
        model.generate(
            "Write a short Python function to add two integers.", max_new_tokens=16
        )
        again = model.generate(prompt, max_new_tokens=16)
        assert (
            first["generated_ids"] and first["generated_ids"] == again["generated_ids"]
        )
        assert first["reused_prompt_tokens"] == again["reused_prompt_tokens"] == 0
        assert (
            model.ready["memory"]["weights_used_bytes"]
            < model.ready["memory"]["device_total_bytes"]
        )
        assert model.tokenize("Hello")
        with pytest.raises(ValueError, match="max_context"):
            model.generate(token_ids=[1] * 512, max_new_tokens=1)
