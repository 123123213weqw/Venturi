"""Prepared decode graph contracts; CUDA cases are explicitly opt-in."""

import os
from types import SimpleNamespace

import pytest
import torch

from neox_reference import tiny_checkpoint
from venturi import load_checkpoint
from venturi.generation import generate_tokens
from venturi.models import GPTNeoXModel
from venturi.models.cuda_graph import GPTNeoXCUDAGraphSession


def test_graph_requires_cuda_fp16():
    for device, dtype in (("cpu", torch.float16), ("cuda:0", torch.float32)):
        with pytest.raises(ValueError, match="CUDA FP16"):
            GPTNeoXCUDAGraphSession(
                SimpleNamespace(device=torch.device(device), dtype=dtype),
                8, decode_start=2, decode_steps=3,
            )


@pytest.mark.parametrize("capacity,start,steps", [
    (8, 0, 1), (8, True, 1), (8, 2, -1), (8, 2, True),
    (300, 2, 257), (8, 7, 2), (True, 2, 1),
])
def test_bad_plan_fails_before_cuda_allocation(capacity, start, steps):
    model = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float16)
    with pytest.raises(ValueError):
        GPTNeoXCUDAGraphSession(model, capacity, decode_start=start, decode_steps=steps)


@pytest.mark.parametrize("lengths", [None, "3", [True], [0], [21], [3, 3], [1.5], list(range(1, 10))])
def test_bad_prefill_plan_fails_before_cuda_allocation(lengths):
    model = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float16)
    with pytest.raises(ValueError, match="prefill_lengths"):
        GPTNeoXCUDAGraphSession(model, 20, decode_start=2, decode_steps=0, prefill_lengths=lengths)


@pytest.fixture
def gpu_model(make_checkpoint):
    if os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip("Set VENTURI_TEST_CUDA=1 for CUDA graph tests")
    if not torch.cuda.is_available():
        pytest.fail("VENTURI_TEST_CUDA is set but CUDA is unavailable")
    config, weights = tiny_checkpoint()
    weights = {name: tensor.half() for name, tensor in weights.items()}
    path = make_checkpoint(config=config, tensors=weights)
    return GPTNeoXModel(load_checkpoint(path, device="cuda:0", loader="safetensors"))


@pytest.mark.cuda
def test_exact_logits_varied_tokens_and_request_isolation(gpu_model):
    graph = gpu_model.create_cuda_graph_session(12, decode_start=3, decode_steps=8)
    eager = gpu_model.create_session(12)
    for prompt in ([2, 5, 9], [11, 17, 3], [2, 5, 9]):
        graph.reset()
        eager.reset()
        torch.testing.assert_close(graph.forward(prompt), eager.forward(prompt), rtol=0, atol=0)
        for token in (2, 5, 3, 11, 17, 0, 30, 7):
            expected = eager.forward([token])
            actual = graph.forward([token])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert graph.select_token(actual) == int(expected.argmax())


@pytest.mark.cuda
def test_greedy_eos_reset_and_coverage(gpu_model):
    graph = gpu_model.create_cuda_graph_session(12, decode_start=3, decode_steps=7)
    expected = generate_tokens(gpu_model, [2, 5, 9], max_new_tokens=8, eos_token_ids=())
    for _ in range(2):
        actual = generate_tokens(
            gpu_model, [2, 5, 9], max_new_tokens=8, eos_token_ids=(), session=graph
        )
        assert actual.generated_ids == expected.generated_ids
    first = expected.generated_ids[0]
    stopped = generate_tokens(
        gpu_model, [2, 5, 9], max_new_tokens=8, eos_token_ids=(first,), session=graph
    )
    assert stopped.generated_ids == (first,) and stopped.finish_reason == "eos"
    for prompt in ([2, 5, 9, 11], [2, 5, 9, 11, 17]):
        a = generate_tokens(gpu_model, prompt, max_new_tokens=4, eos_token_ids=())
        b = generate_tokens(gpu_model, prompt, max_new_tokens=4, eos_token_ids=(), session=graph)
        assert a.generated_ids == b.generated_ids
    with pytest.raises(ValueError, match="prepared CUDA graph"):
        generate_tokens(gpu_model, [2, 5], max_new_tokens=8, session=graph)
    graph.reset()
    graph.forward([2, 5, 9])
    for _ in range(7):
        graph.forward([2])
    with pytest.raises(ValueError, match="not prepared"):
        graph.forward([2])


@pytest.mark.cuda
def test_graph_stream_ownership_and_no_hot_allocator_growth(gpu_model):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph = gpu_model.create_cuda_graph_session(8, decode_start=2, decode_steps=3)
        graph.forward([2, 5])
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        logits = graph.forward([3])
        assert graph.select_token(logits) >= 0
        assert torch.cuda.max_memory_allocated() == before
        assert graph.forward([7], all_logits=True).shape == (1, gpu_model.config.vocab_size)
    with pytest.raises(ValueError, match="owning CUDA stream"):
        graph.forward([2])
    with pytest.raises(ValueError, match="owning CUDA stream"):
        graph.reset()
    with pytest.raises(ValueError, match="owning CUDA stream"):
        graph.select_token(logits)
    stream.synchronize()


@pytest.mark.cuda
def test_captured_nonfinite_logits_fail_and_reset(gpu_model):
    graph = gpu_model.create_cuda_graph_session(8, decode_start=2, decode_steps=3)
    graph.forward([2, 5])
    head = gpu_model._weights["embed_out.weight"]
    original = head.clone()
    head.fill_(float("nan"))  # Deliberate corruption to exercise captured checks.
    logits = graph.forward([3])
    with pytest.raises(RuntimeError, match="nonfinite"):
        graph.select_token(logits)
    with pytest.raises(RuntimeError, match="reset"):
        graph.forward([2])
    head.copy_(original)
    graph.reset()
    graph.forward([2, 5])
    assert graph.select_token(graph.forward([3])) >= 0


@pytest.mark.cuda
def test_prefill_graphs_exact_shapes_fallback_and_cache(gpu_model):
    graph = gpu_model.create_cuda_graph_session(
        12, decode_start=1, decode_steps=11, prefill_lengths=(1, 3, 5)
    )
    eager = gpu_model.create_session(12)
    for prompt in ([2], [2, 5, 9], [11, 7, 3], [2, 5, 9, 11, 17], [2, 5], [2, 5, 9]):
        graph.reset()
        eager.reset()
        actual, expected = graph.forward(prompt), eager.forward(prompt)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert graph.select_token(actual) == int(expected.argmax())
        for chunk in ([17, 3], [11]):
            actual, expected = graph.forward(chunk), eager.forward(chunk)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    graph.reset()
    eager.reset()
    # All-logits prefill must not accidentally return only the captured last row.
    actual = graph.forward([2, 5, 9], all_logits=True)
    expected = eager.forward([2, 5, 9], all_logits=True)
    assert actual.shape == (3, gpu_model.config.vocab_size)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # Queue changed prompts without reading GPU outputs between replays.
    pending = []
    for prompt in ([2, 5, 9], [17, 11, 3], [2, 5, 9]):
        graph.reset()
        pending.append((prompt, graph.forward(prompt).clone()))
    for prompt, actual in pending:
        eager.reset()
        assert torch.equal(actual.view(torch.int16), eager.forward(prompt).view(torch.int16))


@pytest.mark.cuda
def test_prefill_graph_greedy_eos_failure_and_stream(gpu_model):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph = gpu_model.create_cuda_graph_session(
            8, decode_start=3, decode_steps=4, prefill_lengths=(3,)
        )
        expected = generate_tokens(gpu_model, [2, 5, 9], max_new_tokens=5, eos_token_ids=())
        for _ in range(2):
            actual = generate_tokens(gpu_model, [2, 5, 9], max_new_tokens=5, eos_token_ids=(), session=graph)
            assert actual.generated_ids == expected.generated_ids
        actual = generate_tokens(gpu_model, [2, 5, 9], max_new_tokens=5,
                                 eos_token_ids=(expected.generated_ids[0],), session=graph)
        assert actual.generated_ids == expected.generated_ids[:1] and actual.finish_reason == "eos"
        graph.reset()
        # Prefill replay reuses its captured GPU storage, including finite checks.
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        logits = graph.forward([11, 7, 3])
        graph.select_token(logits)
        assert torch.cuda.max_memory_allocated() == before
        head = gpu_model._weights["embed_out.weight"]
        original = head.clone()
        head.fill_(float("nan"))
        graph.reset()
        logits = graph.forward([2, 5, 9])
        with pytest.raises(RuntimeError, match="nonfinite"):
            graph.select_token(logits)
        with pytest.raises(RuntimeError, match="reset"):
            graph.forward([3])
        head.copy_(original)
        graph.reset()
        assert graph.select_token(graph.forward([2, 5, 9])) >= 0
    with pytest.raises(ValueError, match="owning CUDA stream"):
        graph.forward([2, 5, 9])
    stream.synchronize()


@pytest.mark.cuda
def test_prefill_only_plan_without_decode_graphs(gpu_model):
    graph = gpu_model.create_cuda_graph_session(
        8, decode_start=3, decode_steps=0, prefill_lengths=(3,)
    )
    for prompt in ([2, 5, 9], [2, 5]):
        a = generate_tokens(gpu_model, prompt, max_new_tokens=1, eos_token_ids=())
        b = generate_tokens(gpu_model, prompt, max_new_tokens=1, eos_token_ids=(), session=graph)
        assert a.generated_ids == b.generated_ids
