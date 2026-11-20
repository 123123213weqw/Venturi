"""Engine-independent, opt-in GPU admission and coarse telemetry utilities.

Importing this module does not load a model or query a device. Call idle_gpu
explicitly for the existing non-destructive admission check. Busy/controller
samples are not achieved SM occupancy, FLOPS or measured DRAM bandwidth.
"""

import csv
from datetime import datetime
import statistics
import subprocess


FIELDS = [
    "timestamp",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
]


def telemetry(path, waves):
    rows = []
    for row in csv.reader(path.read_text().splitlines()):
        if len(row) != len(FIELDS):
            continue
        try:
            stamp = datetime.strptime(
                row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
        except ValueError:
            continue
        values = {
            name: float(value) if value.strip() not in ("N/A", "[N/A]") else None
            for name, value in zip(FIELDS[1:], row[1:])
        }
        rows.append(dict(timestamp=stamp, **values))
    summaries = {}
    for mode in ("single", "parallel"):
        chosen = [
            row
            for row in rows
            if any(
                w["mode"] == mode
                and w["start_unix"] + 1 <= row["timestamp"] <= w["end_unix"]
                for w in waves
            )
        ]
        summaries[mode] = dict(samples=len(chosen))
        for name in FIELDS[1:]:
            values = [r[name] for r in chosen if r[name] is not None]
            summaries[mode][name] = (
                dict(
                    mean=statistics.mean(values),
                    minimum=min(values),
                    maximum=max(values),
                )
                if values
                else None
            )
    return summaries


def idle_gpu():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        if (
            line.strip()
            and line.split(",")[1].strip() != "/usr/libexec/gnome-remote-desktop-daemon"
        ):
            raise RuntimeError(f"GPU is occupied; no workloads were killed: {line}")
    return output
