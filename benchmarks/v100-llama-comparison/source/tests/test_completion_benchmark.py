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
    return dict(prompt=[1,2,3],n_predict=32,id_slot=0,temperature=0,stream=False,ignore_eos=True,cache_prompt=False)


def test_fixed_native_request():
    assert load("serve_native_benchmark").request_values(request(),8,256,128) == ([1,2,3],32,0)


@pytest.mark.parametrize("patch", [dict(prompt=[]),dict(prompt=[True]),dict(prompt=[128]),dict(prompt=[1]*33),dict(n_predict=True),dict(n_predict=129),dict(id_slot=8),dict(ignore_eos=False),dict(cache_prompt=True),dict(stream=True),dict(temperature=1),dict(repeat_penalty=1.1)])
def test_reject_changed_benchmark_work(patch):
    body = request() | patch
    with pytest.raises(ValueError):
        load("serve_native_benchmark").request_values(body,8,256,128)


def test_reject_nonobject_request():
    with pytest.raises(ValueError):
        load("serve_native_benchmark").request_values([],8,256,128)


def test_aggregate_uses_global_wall_and_no_fake_parallel_decode():
    wave = dict(request_count=8,emitted_tokens=256,wall_seconds=10,server_cpu_core_seconds=6,requests=[dict(client_completion_seconds=8,response={}) for _ in range(8)])
    result, = load("benchmark_completion_api").summarize([wave])
    assert result["aggregate_complete_request_output_tps"] == 25.6
    assert result["mean_server_cpu_core_seconds"] == 6
    assert "native_single_post_first_tps" not in result


def test_single_decode_subtracts_same_call_first_token():
    wave = dict(request_count=1,emitted_tokens=32,wall_seconds=3,server_cpu_core_seconds=2,requests=[dict(client_completion_seconds=3,response=dict(native_timings=dict(native_generation_seconds=2,native_time_to_first_token_seconds=1)))])
    result, = load("benchmark_completion_api").summarize([wave])
    assert result["native_single_post_first_tps"] == 31
    assert result["aggregate_complete_request_output_tps"] == 32/3


def test_llama_retained_kv_is_not_prompt_reuse():
    result = dict(tokens=[4,5],tokens_predicted=2,tokens_evaluated=3,tokens_cached=4,timings=dict(prompt_n=3))
    load("benchmark_completion_api").validate_response(result,[1,2,3],[4,5],False)
    with pytest.raises(AssertionError):
        load("benchmark_completion_api").validate_response(result | dict(timings=dict(prompt_n=1)),[1,2,3],[4,5],False)
