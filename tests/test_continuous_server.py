"""Explicit loopback continuous admission, without a model/device dependency."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from test_completion_benchmark import load, request
from test_continuous_scheduler import Batch, reference


def adapter(batch, policy="round-robin"):
    return load("serve_native_benchmark").ContinuousCompletionBatcher(
        batch, token_budget=8, prefill_chunk_size=3, max_pending=64,
        prefill_policy=policy,
    )


@pytest.mark.parametrize("policy", ["round-robin", "balanced", "balanced-short"])
def test_concurrent_http_lanes_use_private_dynamic_slots_and_own_timing_scope(policy):
    batch = Batch(capacities=(64,) * 2)
    service = adapter(batch, policy)
    try:
        def run(i):
            prompt = [i % 31, (i + 1) % 31]
            result = service.submit(i, prompt, i % 6 + 1).result(timeout=3)
            assert tuple(result["tokens"]) == reference(prompt, i % 6 + 1)
            assert result["id_slot"] == i  # Not a native slot (only two exist).
            assert result["tokens_evaluated"] == 2
            assert result["native_timings"] is None
            assert "not native-only" in result["scheduler_timings_scope"]
            timing = result["scheduler_timings"]
            assert 0 <= timing["queue_seconds"] <= timing["time_to_first_token_seconds"] <= timing["wall_seconds"]
            return result["request_id"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(run, range(32)))
        assert len(set(ids)) == 32
    finally:
        service.close()
    assert not service.worker.info()["alive"]
    with pytest.raises(RuntimeError, match="closed"):
        service.submit(0, [1], 1)


def test_response_cancellation_releases_only_its_lease_at_safe_boundary():
    batch = Batch(capacities=(64,))
    entered, release = threading.Event(), threading.Event()
    batch.hook = lambda: (entered.set(), release.wait(timeout=3))
    service = adapter(batch)
    try:
        abandoned = service.submit(6, [1, 2], 8)
        assert entered.wait(timeout=3)
        assert abandoned.cancel()
        fresh = service.submit(7, [4, 5], 3)
        release.set()
        result = fresh.result(timeout=3)
        assert tuple(result["tokens"]) == reference([4, 5], 3)
        assert abandoned.cancelled() and result["id_slot"] == 7
        assert service.worker.info()["scheduling"]["cancelled_requests"] == 1
    finally:
        release.set()
        service.close()


def test_worker_shutdown_cancels_bridge_and_failure_propagates_without_fallback():
    batch = Batch()
    entered, release = threading.Event(), threading.Event()
    batch.hook = lambda: (entered.set(), release.wait(timeout=3))
    service = adapter(batch)
    response = service.submit(0, [1], 8)
    assert entered.wait(timeout=3)
    with pytest.raises(TimeoutError):
        service.worker.close(timeout=0)
    release.set()
    service.close()
    assert response.cancelled()

    batch = Batch()
    batch.fail_step = True
    service = adapter(batch)
    response = service.submit(0, [1], 8)
    with pytest.raises(RuntimeError, match="step fault"):
        response.result(timeout=3)
    with pytest.raises(RuntimeError, match="failure"):
        service.close()
    assert not service.worker.info()["alive"]


def test_only_explicit_continuous_validation_allows_long_chunked_prompts():
    module = load("serve_native_benchmark")
    body = request() | dict(prompt=[1] * 80)
    with pytest.raises(ValueError):
        module.request_values(body, 8, 128, 31)
    assert len(module.request_values(body, 8, 128, 31, max_prompt_tokens=128)[0]) == 80
    with pytest.raises(ValueError):
        module.request_values(body, 8, 100, 31, max_prompt_tokens=128)


def test_http_benchmark_does_not_treat_scheduler_timing_as_native_only_decode():
    response = dict(native_timings=None, scheduler_timings=dict(
        time_to_first_token_seconds=0.3, max_inter_token_seconds=0.2,
    ))
    waves = [dict(request_count=1, emitted_tokens=4, wall_seconds=2.0,
                  server_cpu_core_seconds=1.2,
                  requests=[dict(client_completion_seconds=1.9, response=response)])]
    result = load("benchmark_completion_api").summarize(waves)[0]
    assert result["aggregate_complete_request_output_tps"] == 2.0
    assert "native_single_post_first_tps" not in result
    assert result["mean_scheduler_token_availability_ttft_seconds"] == 0.3
    assert "not HTTP streaming/native-only" in result["scheduler_timing_scope"]


@pytest.mark.parametrize("flags", [
    ["--mode", "pp"], ["--batch-target"], ["--batch-mtp"],
    ["--mtp-artifact", "missing"], ["--adaptive-mtp"], ["--target-layer-graph"],
    ["--mtp-depth", "2"], ["--prefill-chunk-size", "0"],
    ["--prefill-chunk-size", "33"], ["--max-pending", "0"],
    ["--max-pending", "65537"],
])
def test_unsupported_continuous_combinations_rejected_before_model_or_gpu(flags, tmp_path):
    tool = Path(__file__).resolve().parents[1] / "tools/serve_native_benchmark.py"
    result = subprocess.run([
        sys.executable, str(tool), "missing-model", "--library", "missing-library",
        "--mode", "tp", "--port", "18033", "--ready-file", str(tmp_path / "ready.json"),
        "--continuous-target", *flags,
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "Continuous target requires" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "ready.json").exists()
