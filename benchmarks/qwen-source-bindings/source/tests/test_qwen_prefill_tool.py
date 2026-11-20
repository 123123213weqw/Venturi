"""Prefill accounting and deterministic fixtures, without starting a model."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from benchmark_qwen_prefill import make_prompts, summarize  # noqa: E402


def test_fixture_has_exact_raw_token_lengths_and_suffix():
    encode = lambda text: list(text.encode())  # noqa: E731
    prompts = make_prompts(encode, [128, 509])
    assert {n: len(ids) for n, ids in prompts.items()} == {128: 128, 509: 509}
    suffix = encode("\nContinue the explanation:")
    assert all(ids[-len(suffix) :] == suffix for ids in prompts.values())
    assert prompts == make_prompts(encode, [128, 509])
    with pytest.raises(ValueError, match="empty"):
        make_prompts(lambda text: [], [128])
    with pytest.raises(ValueError, match="prefix/suffix"):
        make_prompts(encode, [1])


def test_prefill_token_rate_uses_only_prefill_interval_and_cpu_uses_client_wall():
    runs = [
        dict(
            prompt_length=512,
            wall_seconds=wall,
            native_cpu_seconds=wall / 10,
            python_cpu_seconds=0.01,
            result={
                "timings": {
                    "prefill_seconds": prefill,
                    "first_token_seconds": prefill + 1,
                    "decode_seconds": 0.25,
                }
            },
        )
        for wall, prefill in [(2, 1), (4, 3)]
    ]
    report = summarize(runs)["512"]
    assert report["native_prefill_tokens_per_second"] == 256
    assert report["median_native_prefill_seconds"] == 2
    assert report["median_native_first_token_seconds"] == 3
    assert report["median_native_decode_seconds"] == 0.25
    assert report["client_wall_seconds"] == 6
    assert report["native_cpu_percent"] == pytest.approx(10)
