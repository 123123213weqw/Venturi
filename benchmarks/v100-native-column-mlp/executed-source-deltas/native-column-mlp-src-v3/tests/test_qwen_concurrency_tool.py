"""Research metric accounting: concurrent latency must not be summed as wall time."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from benchmark_qwen_concurrency import build_batches, compare_ids, summarize  # noqa: E402


def test_aggregate_rate_uses_shared_wall_and_actual_decode_rows():
    host = {
        k: 0
        for k in (
            "engine_boundary_ns",
            "program_submit_ns",
            "program_post_ns",
            "engine_commit_output_ns",
            "engine_maintenance_ns",
            "device_wait_ns",
        )
    }
    before = dict(decode_rounds=10, decode_row_rounds=20, host_work=host)
    after = copy.deepcopy(before)
    after.update(decode_rounds=20, decode_row_rounds=60)
    after["host_work"].update(
        program_submit_ns=100_000_000, device_wait_ns=1_800_000_000
    )
    row = dict(
        generated_ids=[1] * 11,
        reused_prompt_tokens=0,
        timings=dict(
            total_seconds=2.0,
            generation_wall_seconds=1.0,
            first_token_seconds=0.5,
            prompt_wall_seconds=0.4,
        ),
        engine_timing=dict(queue_wait_seconds=0.1),
    )
    result = summarize(
        dict(
            results=[row] * 4,
            stats_before=before,
            stats_after=after,
            wall_seconds=2.0,
            memory={},
        )
    )
    assert result["aggregate_output_tps"] == 22
    assert result["mean_actual_decode_batch"] == 4
    assert result["per_request_generation_tps"] == [10] * 4
    assert result["host_active_fraction_of_batch_wall"] == pytest.approx(0.05)
    assert result["device_wait_seconds"] == pytest.approx(1.8)


@pytest.mark.parametrize(
    "a,b,equal,prefix",
    [
        ([1, 2], [1, 2], True, 2),
        ([1, 2], [1, 3], False, 1),
        ([1], [1, 2], False, 1),
        ([], [], True, 0),
    ],
)
def test_comparison_does_not_hide_batch_numerical_divergence(a, b, equal, prefix):
    result = compare_ids(dict(generated_ids=a), dict(generated_ids=b))
    assert result["equal"] is equal
    assert result["matching_prefix"] == prefix


def test_fixed_workload_is_equal_size_and_reuse_disabled():
    result = build_batches([dict(name="long-512", ids=list(range(512)))])
    requests = result["requests"]
    assert len(requests) == 8
    assert all(
        len(r["token_ids"]) == 512 and r["max_new_tokens"] == 64 for r in requests
    )
    assert all(r["ignore_eos"] and not r["reuse"] for r in requests)
    assert len({tuple(r["token_ids"]) for r in requests}) == 8


def test_commit_timing_does_not_invent_intra_speculation_intervals():
    from benchmark_qwen_concurrency import commit_intervals

    result = commit_intervals(
        [
            dict(tokens=1, generation_ns=0),
            dict(tokens=2, generation_ns=20_000_000),
            dict(tokens=6, generation_ns=40_000_000),
            dict(tokens=7, generation_ns=290_000_000),
        ]
    )
    assert result["single_token_intervals"] == 2
    assert result["max_ms"] == 250
    assert result["intervals_over_100ms"] == 1
    with pytest.raises(ValueError):
        commit_intervals(
            [dict(tokens=2, generation_ns=1), dict(tokens=1, generation_ns=2)]
        )


def test_disabled_cache_cli_selects_root_only_reservations(tmp_path, monkeypatch):
    import benchmark_qwen_concurrency as tool

    workloads = tmp_path / "workloads.json"
    workloads.write_text(
        __import__("json").dumps([dict(name="long-512", ids=list(range(512)))])
    )
    runner = tmp_path / "probe"
    runner.write_text("fake")
    configs = []

    def execute(args, out, name, config, batches):
        configs.append(config)
        return dict(ok=True)

    monkeypatch.setattr(tool, "execute", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "/explicit/model",
            "--runner",
            str(runner),
            "--workloads",
            str(workloads),
            "--output-dir",
            str(tmp_path / "out"),
            "--only",
            "cache-disabled",
        ],
    )
    tool.main()
    assert len(configs) == 1
    assert configs[0]["cache"] is False
    assert configs[0]["checkpoint_slots"] == 0
