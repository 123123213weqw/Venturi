"""GPU busy samples are coarse whole-device telemetry, not occupancy counters."""

from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from gpu_telemetry import telemetry  # noqa: E402


def test_wave_filter_excludes_boundary_averaging_and_preserves_metric_names(tmp_path):
    start = datetime(2026, 10, 16, 12)
    samples = []
    for offset, busy in ((0.5, 1), (1.5, 90), (2.5, 100), (6.5, 99), (10, 0)):
        stamp = (start + timedelta(seconds=offset)).strftime("%Y/%m/%d %H:%M:%S.%f")[
            :-3
        ]
        samples.append(f"{stamp}, {busy}, 50, 16000, 200, 50, 2500, 10000")
    path = tmp_path / "telemetry.csv"
    path.write_text("timestamp,header\n" + "\n".join(samples))
    results = telemetry(
        path,
        [
            dict(
                mode="single",
                start_unix=start.timestamp(),
                end_unix=start.timestamp() + 3,
            ),
            dict(
                mode="parallel",
                start_unix=start.timestamp() + 5,
                end_unix=start.timestamp() + 8,
            ),
        ],
    )
    assert results["single"]["samples"] == 2
    assert results["single"]["utilization.gpu"]["mean"] == 95
    assert results["single"]["utilization.memory"]["mean"] == 50
    assert results["parallel"]["utilization.gpu"]["mean"] == 99
    assert "occupancy" not in results["single"]


def test_missing_telemetry_is_not_reported_as_zero_occupancy_or_power(tmp_path):
    start = datetime(2026, 10, 16, 12)
    path = tmp_path / "telemetry.csv"
    path.write_text("2026/10/16 12:00:01.500, 100, 60, 16000, [N/A], 50, 2500, 10000\n")
    result = telemetry(
        path,
        [
            dict(
                mode="single",
                start_unix=start.timestamp(),
                end_unix=start.timestamp() + 3,
            )
        ],
    )
    assert result["single"]["power.draw"] is None
    assert result["parallel"]["samples"] == 0
    assert result["parallel"]["utilization.gpu"] is None
