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


@pytest.mark.cuda
@pytest.mark.parametrize("kv_dtype", ["bf16", "k8v4"])
def test_real_native_single_parallel_switch_and_state_isolation(checkpoint, kv_dtype):
    runner = os.environ.get("VENTURI_QWEN_RUNNER")
    if os.environ.get("VENTURI_TEST_CUDA") != "1" or not runner:
        pytest.skip("Set VENTURI_TEST_CUDA=1 and VENTURI_QWEN_RUNNER")
    prompts = [
        "The capital of France is",
        "def add(a, b):\n    return",
        "The sequence 2, 4, 6, 8 continues with",
        "Once upon a time in a quiet village",
    ] * 2
    with QwenNativeSession(
        checkpoint,
        runner=runner,
        max_context=512,
        max_concurrency=4,
        prefill_chunk=128,
        kv_dtype=kv_dtype,
        timeout=600,
    ) as model:
        process = model._process.pid
        kwargs = dict(max_new_tokens=64, raw_prompt=True, ignore_eos=True)
        serial = model.generate_many(prompts, **kwargs)
        assert serial["mode"] == "single"
        assert serial["runtime"]["mean_decode_batch"] == 1
        model.set_mode("parallel")
        parallel = model.generate_many(prompts, **kwargs)
        assert parallel["runtime"]["mean_decode_batch"] > 1
        assert parallel["runtime"]["mean_decode_batch"] <= 4
        # Different compact batch shapes can change quantized floating-point
        # arithmetic and near-tied greedy choices; cross-B IDs are not an oracle.
        one_row = model.generate_many([prompts[0]], **kwargs)
        assert (
            one_row["results"][0]["generated_ids"]
            == serial["results"][0]["generated_ids"]
        )
        assert all(r["reused_prompt_tokens"] == 0 for r in parallel["results"])
        assert all(len(r["generated_ids"]) == 64 for r in parallel["results"])
        assert model._process.pid == process
        model.set_mode("single")
        assert (
            model.generate(prompts[0], **kwargs)["generated_ids"]
            == serial["results"][0]["generated_ids"]
        )
        # Varying EOS lengths must not leak state between rows or reorder output.
        chat = [
            "Reply only with YES.",
            "What is 2+2?",
            "Explain gravity in one sentence.",
        ]
        a = model.generate_many(chat, max_new_tokens=32)
        b = model.generate_many(chat, mode="parallel", max_new_tokens=32)
        assert [r["index"] for r in b["results"]] == list(range(len(chat)))
        assert all(0 < len(r["generated_ids"]) <= 32 for r in b["results"])
        oracle = model.generate_many(
            [
                "Reply with exactly the word APPLE and nothing else.",
                "Reply with exactly the word RIVER and nothing else.",
                "Reply with exactly the word CLOUD and nothing else.",
                "Reply with exactly the word MUSIC and nothing else.",
            ],
            mode="parallel",
            max_new_tokens=16,
        )
        assert [r["text"].strip() for r in oracle["results"]] == [
            "APPLE",
            "RIVER",
            "CLOUD",
            "MUSIC",
        ]
        # Every prompt is validated before any request in the list executes.
        with pytest.raises(ValueError, match="max_context"):
            model.generate_many(["hello", "word " * 512], max_new_tokens=32)
        assert (
            model.generate("What is 2+2?", max_new_tokens=32)["generated_ids"]
            == a["results"][1]["generated_ids"]
        )
        bounded = model.generate_many(["hello"] * 16, mode="parallel", max_new_tokens=2)
        assert len(bounded["results"]) == 16


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kv,graphs,spec",
    [
        ("bf16", True, "none"),
        ("k8v4", False, "none"),
        ("bf16", True, "dflash2"),
    ],
)
def test_real_cpu_wait_preserves_single_request_execution(checkpoint, kv, graphs, spec):
    runner = os.environ.get("VENTURI_QWEN_RUNNER")
    if os.environ.get("VENTURI_TEST_CUDA") != "1" or not runner:
        pytest.skip("Set VENTURI_TEST_CUDA=1 and VENTURI_QWEN_RUNNER")
    outputs = []
    for policy, expected_flags in (("auto", 0), ("blocking", 4)):
        with QwenNativeSession(
            checkpoint,
            runner=runner,
            max_context=512,
            prefill_chunk=128,
            cpu_wait=policy,
            kv_dtype=kv,
            cuda_graph=graphs,
            speculative=spec,
            timeout=600,
        ) as model:
            assert model.ready["cpu_wait"] == policy
            assert model.ready["cuda_device_flags"] & 7 == expected_flags
            result = model.generate(
                "Explain why the sky looks blue.", max_new_tokens=32
            )
            model.generate("What is 2+2?", max_new_tokens=16)
            replay = model.generate(
                "Explain why the sky looks blue.", max_new_tokens=32
            )
            assert result["generated_ids"] == replay["generated_ids"]
            outputs.append(result["generated_ids"])
    assert outputs[0] == outputs[1]
