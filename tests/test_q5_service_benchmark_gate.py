"""The HTTP driver must not publish 7+1 admission as an eight-row comparison."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "q5_service_benchmark",
    Path(__file__).parents[1] / "tools/benchmark_qwen_q5_service.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def responses(count, cohort=1, reported=None):
    return [
        dict(
            response=dict(
                batch_cohort_id=cohort,
                batch_request_count=count if reported is None else reported,
            )
        )
        for _ in range(count)
    ]


def test_complete_eight_row_wave_passes():
    module.validate_cohort(responses(8), 8)


@pytest.mark.parametrize(
    "values",
    [
        responses(7, reported=7) + responses(1, cohort=2),
        responses(7, reported=8) + responses(1, cohort=2, reported=8),
        responses(7),
    ],
)
def test_partial_or_split_cohorts_are_rejected(values):
    with pytest.raises(AssertionError):
        module.validate_cohort(values, 8)
