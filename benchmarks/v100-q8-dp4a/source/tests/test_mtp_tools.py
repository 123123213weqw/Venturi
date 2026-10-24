"""Acceptance denominators and cohort throughput must not count requests twice."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(path))
    spec = spec_from_file_location("mtp_bench", path / "benchmark_qwen_mtp.py")
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mtp_acceptance_uses_all_proposals(monkeypatch):
    m = load(monkeypatch)
    rows = [
        dict(
            proposed_tokens=10,
            accepted_tokens=9,
            verification_calls=10,
            replayed_input_tokens=1,
        ),
        dict(
            proposed_tokens=2,
            accepted_tokens=0,
            verification_calls=2,
            replayed_input_tokens=2,
        ),
    ]
    r = m.acceptance(rows)
    assert r["acceptance_rate"] == 9 / 12 and r["proposed_by_depth"] == [12]
    assert m.acceptance([])["acceptance_rate"] is None


def test_mtp_throughput_uses_global_wall_no_fake_concurrent_decode(monkeypatch):
    m = load(monkeypatch)
    wave = dict(
        emitted_tokens=8,
        wall_seconds=2.0,
        process_cpu_seconds=3.0,
        requests=[
            dict(completion_latency_seconds=1.0),
            dict(completion_latency_seconds=2.0),
        ],
    )
    row = dict(
        request_count=2,
        variant="baseline",
        cohort=wave,
        native_timings=[
            dict(native_time_to_first_token_seconds=0.1, native_generation_seconds=1.0),
            dict(native_time_to_first_token_seconds=0.2, native_generation_seconds=2.0),
        ],
    )
    r = m.summarize([row])[0]
    assert r["aggregate_complete_request_output_tps"] == 4
    assert r["mean_request_completion_seconds"] == 1.5
    assert "single_request_native_post_first_output_tps" not in r


def test_mtp_reference_comparison_has_cosine_and_zero_case(monkeypatch):
    import torch

    path = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(path))
    spec = spec_from_file_location("mtp_qualifier", path / "qualify_qwen_mtp.py")
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    r = m.block_comparison(x, x.double())
    assert r["centered_cosine"] == 1 and r["relative_l2"] == 0
    assert (
        m.block_comparison(torch.zeros(2, 3), torch.zeros(2, 3))["centered_cosine"] == 1
    )


def test_verification_microbenchmark_includes_two_token_head(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(path))
    spec = spec_from_file_location("q8_pairs", path / "benchmark_q8_tiles.py")
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.token_counts("text/output_head", False, True) == (1, 2)
    assert m.token_counts("text/mlp/down", False, True) == (1, 2, 4, 26)
    assert m.token_counts("text/output_head", False, False) == (1,)
    assert m.token_counts("text/mlp/down", True, True) == (1,)
