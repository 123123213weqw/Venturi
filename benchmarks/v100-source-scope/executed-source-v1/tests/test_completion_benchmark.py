"""Benchmark adapters reject changed work and keep aggregate timing honest."""

import importlib.util
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request():
    return dict(
        prompt=[1, 2, 3],
        n_predict=32,
        id_slot=0,
        temperature=0,
        stream=False,
        ignore_eos=True,
        cache_prompt=False,
    )


def test_fixed_native_request():
    assert load("serve_native_benchmark").request_values(request(), 8, 256, 128) == (
        [1, 2, 3],
        32,
        0,
    )


@pytest.mark.parametrize(
    "patch",
    [
        dict(prompt=[]),
        dict(prompt=[True]),
        dict(prompt=[128]),
        dict(prompt=[1] * 33),
        dict(n_predict=True),
        dict(n_predict=129),
        dict(id_slot=8),
        dict(ignore_eos=False),
        dict(cache_prompt=True),
        dict(stream=True),
        dict(temperature=1),
        dict(repeat_penalty=1.1),
    ],
)
def test_reject_changed_benchmark_work(patch):
    body = request() | patch
    with pytest.raises(ValueError):
        load("serve_native_benchmark").request_values(body, 8, 256, 128)


def test_reject_nonobject_request():
    with pytest.raises(ValueError):
        load("serve_native_benchmark").request_values([], 8, 256, 128)


def test_aggregate_uses_global_wall_and_no_fake_parallel_decode():
    wave = dict(
        request_count=8,
        emitted_tokens=256,
        wall_seconds=10,
        server_cpu_core_seconds=6,
        requests=[dict(client_completion_seconds=8, response={}) for _ in range(8)],
    )
    (result,) = load("benchmark_completion_api").summarize([wave])
    assert result["aggregate_complete_request_output_tps"] == 25.6
    assert result["mean_server_cpu_core_seconds"] == 6
    assert "native_single_post_first_tps" not in result


def test_single_decode_subtracts_same_call_first_token():
    wave = dict(
        request_count=1,
        emitted_tokens=32,
        wall_seconds=3,
        server_cpu_core_seconds=2,
        requests=[
            dict(
                client_completion_seconds=3,
                response=dict(
                    native_timings=dict(
                        native_generation_seconds=2,
                        native_time_to_first_token_seconds=1,
                    )
                ),
            )
        ],
    )
    (result,) = load("benchmark_completion_api").summarize([wave])
    assert result["native_single_post_first_tps"] == 31
    assert result["aggregate_complete_request_output_tps"] == 32 / 3


def test_llama_retained_kv_is_not_prompt_reuse():
    result = dict(
        tokens=[4, 5],
        tokens_predicted=2,
        tokens_evaluated=3,
        tokens_cached=4,
        timings=dict(prompt_n=3),
    )
    load("benchmark_completion_api").validate_response(result, [1, 2, 3], [4, 5], False)
    with pytest.raises(AssertionError):
        load("benchmark_completion_api").validate_response(
            result | dict(timings=dict(prompt_n=1)), [1, 2, 3], [4, 5], False
        )


class FakeBatch:
    def __init__(self):
        self.calls = []
        self.fail = False

    def reset(self, *, slots):
        self.reset_slots = slots

    def generate(self, prompts, *, max_new_tokens, eos_token_ids, slots):
        if self.fail:
            raise RuntimeError("injected cohort failure")
        self.calls.append((prompts, max_new_tokens, slots))
        return [[prompt[0]] * n for prompt, n in zip(prompts, max_new_tokens)]

    def info(self):
        return dict(
            requests=[
                dict(
                    native_time_to_first_token_seconds=0.1,
                    native_generation_seconds=0.2,
                )
                for _ in range(8)
            ],
            **{
                k: 1
                for k in (
                    "group_forward_calls",
                    "input_rows",
                    "decode_group_calls",
                    "prefill_group_calls",
                    "head_calls",
                    "head_rows",
                    "max_group_rows",
                )
            },
        )


def test_admission_forms_one_cohort_and_preserves_request_identity():
    from concurrent.futures import ThreadPoolExecutor
    import threading

    fake = FakeBatch()
    admission = load("serve_native_benchmark").CompletionBatcher(fake, 8, 0.05)
    barrier = threading.Barrier(8)
    try:

        def run(i):
            barrier.wait()
            return admission.submit(i, [i, 1], i + 1).result(timeout=2)

        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(run, range(8)))
        assert len(fake.calls) == 1
        assert all(
            r["tokens"] == [i] * (i + 1)
            and r["id_slot"] == i
            and r["batch_request_count"] == 8
            for i, r in enumerate(rows)
        )
        assert {r["batch_cohort_id"] for r in rows} == {1}
    finally:
        admission.close()
    assert not admission.thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        admission.submit(0, [1], 2)


def test_admission_fans_out_failure_and_can_process_next_cohort():
    fake = FakeBatch()
    admission = load("serve_native_benchmark").CompletionBatcher(fake, 8, 0.02)
    try:
        fake.fail = True
        futures = [admission.submit(i, [i], 2) for i in range(3)]
        for future in futures:
            with pytest.raises(RuntimeError, match="cohort failure"):
                future.result(timeout=2)
        fake.fail = False
        assert admission.submit(0, [1], 2).result(timeout=2)["tokens"] == [1, 1]
    finally:
        admission.close()
