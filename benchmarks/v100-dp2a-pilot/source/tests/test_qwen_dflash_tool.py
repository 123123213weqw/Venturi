"""DFlash concurrency metrics use shared elapsed time and retain mismatches."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from benchmark_qwen_dflash import ORDER, PROMPTS, compare, summarize  # noqa: E402


def test_finite_abba_configuration_has_eight_distinct_requests():
    assert len(PROMPTS) == len(set(PROMPTS)) == 8
    assert [mode for _, mode in ORDER] == ["none", "dflash2", "dflash2", "none"]


def test_shared_wall_not_summed_request_time_and_acceptance_is_draft_only():
    row = dict(
        generated_ids=[1, 2, 3, 4],
        timings=dict(first_token_seconds=0.1, total_seconds=3),
        speculative=dict(
            rounds=1,
            drafted_tokens=3,
            accepted_tokens=2,
            accepted_per_position=[1, 1, 0, 0, 0, 0, 0, 0],
            fallback_steps=0,
        ),
    )
    wave = dict(
        wall_seconds=2,
        native_cpu_seconds=0.1,
        python_cpu_seconds=0.02,
        result=dict(
            results=[row] * 8, runtime=dict(decode_rounds=1, decode_row_rounds=8)
        ),
    )
    result = summarize([wave, wave])
    assert result["output_tokens"] == 64
    assert result["aggregate_output_tokens_per_second"] == 16
    assert result["native_cpu_percent"] == 5
    assert result["mean_actual_decode_batch"] == 8
    assert result["accepted_draft_fraction"] == pytest.approx(2 / 3)
    assert result["accepted_drafts_per_round"] == 2
    assert result["accepted_per_position"] == [16, 16, 0, 0, 0, 0, 0, 0]
    assert result["median_native_request_total_seconds"] == 3


def test_output_comparison_keeps_first_difference_and_length_difference():
    assert compare([1, 2, 3], [1, 9, 3])["matching_prefix"] == 1
    short = compare([1, 2, 3], [1, 2])
    assert short["equal"] is False
    assert short["matching_prefix"] == 2
    assert short["reference_length"] == 3
    assert short["actual_length"] == 2
