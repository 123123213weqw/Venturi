"""Sample only declared operator waves; never label this as model utilization."""

from pathlib import Path
import csv
import datetime
import json
import statistics

ROOT = Path(__file__).resolve().parent
FIELDS = [
    "gpu_busy_percent",
    "memory_controller_busy_percent",
    "device_used_mib",
    "power_watts",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
]


def main():
    result = {}
    offset = json.loads(
        (ROOT / "v100/staged-campaign-v7/campaign-report.json").read_text()
    )["utc_offset_seconds"]
    for folder in ["study-v1", "staged-study-v5"]:
        base = ROOT / "v100" / folder
        report = json.loads((base / "operators/report.json").read_text())
        rows = list(
            csv.reader((base / "operators/gpu-samples.csv").read_text().splitlines())
        )
        result[folder] = {}
        for arm in sorted({w["arm"] for c in report["cases"] for w in c["waves"]}):
            intervals = [
                (w["start_unix"], w["end_unix"])
                for c in report["cases"]
                for w in c["waves"]
                if w["arm"] == arm
            ]
            selected = []
            for row in rows:
                try:
                    timestamp = (
                        datetime.datetime.strptime(
                            row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
                        )
                        .replace(tzinfo=datetime.timezone.utc)
                        .timestamp()
                        - offset
                    )
                    values = [float(v) for v in row[1:]]
                except ValueError:
                    continue
                if len(values) == 7 and any(a <= timestamp <= b for a, b in intervals):
                    selected.append(values)
            result[folder][arm] = dict(
                samples=len(selected),
                physical_gpu=1,
                values={
                    name: dict(
                        mean=statistics.mean(v[i] for v in selected),
                        min=min(v[i] for v in selected),
                        max=max(v[i] for v in selected),
                    )
                    for i, name in enumerate(FIELDS)
                }
                if selected
                else {},
                scope="Samples inside microbenchmark wave intervals; not model telemetry, achieved occupancy or bandwidth. Some short waves have no samples.",
            )
    (ROOT / "operator-telemetry.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
