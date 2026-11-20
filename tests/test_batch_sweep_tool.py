"""Fixed-cohort sweep accounting; no CUDA initialization or model is needed."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).parents[1] / "benchmarks/v100-batch-sweep/benchmark.py"
SPEC = importlib.util.spec_from_file_location("batch_sweep", PATH)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


def test_help_does_not_load_cuda_or_model():
    result = subprocess.run([sys.executable, str(PATH), "--help"], capture_output=True, text=True, check=True)
    assert "--count {1,2,4,8,16,32}" in result.stdout


def test_prompt_extension_is_explicit_and_request_owned():
    prompts = [[i] * 16 for i in range(8)]
    rows = SWEEP.repeat_prompts(prompts, 32)
    assert rows == prompts * 4
    rows[8][0] = 999
    assert rows[0][0] == prompts[0][0] == 0


def test_nvml_and_pytorch_uuid_display_forms_select_the_same_hardware():
    SWEEP.verify_gpu_uuids(SWEEP.UUIDS)
    SWEEP.verify_gpu_uuids([value.removeprefix("GPU-") for value in SWEEP.UUIDS])
    with pytest.raises(AssertionError):
        SWEEP.verify_gpu_uuids(list(reversed(SWEEP.UUIDS)))
    with pytest.raises(AssertionError):
        SWEEP.verify_gpu_uuids([SWEEP.UUIDS[0], "wrong-device"])


@pytest.mark.parametrize("count", [0, 3, 33, 64])
def test_unsupported_native_count_is_rejected(count):
    with pytest.raises(ValueError):
        SWEEP.repeat_prompts([[i] * 16 for i in range(8)], count)


def test_aggregate_denominator_is_cohort_wall_not_request_time_sum():
    trials = [dict(
        ttft=dict(wall_seconds=0.25),
        decode=dict(wall_seconds=2.0, process_cpu_seconds=2.5, output_tokens=248),
        complete=dict(wall_seconds=2.5, output_tokens=256),
    )]
    value = SWEEP.summarize(trials, 8, 32)
    assert value["decode_aggregate_tps_mean"] == 124
    assert value["decode_per_request_tps_mean"] == 15.5
    assert value["mean_intertoken_ms"] == pytest.approx(2000 / 31)
    assert value["complete_aggregate_tps_mean"] == 102.4
    assert value["decode_process_cpu_cores_mean"] == 1.25


def test_incorrect_token_denominator_is_not_silently_scored():
    with pytest.raises(ValueError):
        SWEEP.summarize([dict(ttft={}, decode=dict(wall_seconds=1, output_tokens=32))], 8, 32)
