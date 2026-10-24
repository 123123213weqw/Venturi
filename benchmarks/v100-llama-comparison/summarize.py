"""Summarize the initial same-source API screen, preserving partial/failed cells."""

import ast
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def telemetry(directory, benchmark, count, gpu, offset):
    windows = [(w["start_unix"], w["end_unix"]) for w in benchmark["waves"] if w["request_count"] == count]
    samples = []
    tz = timezone(timedelta(seconds=offset))
    for row in csv.reader((directory/f"gpu-{gpu}-samples.csv").read_text().splitlines()):
        if len(row) != 8:
            continue
        stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
        if any(a <= stamp <= b for a,b in windows):
            samples.append([float(x.strip()) for x in row[1:]])
    if not samples:
        return dict(samples=0, scope="No sample inside this short measured window")
    names = ("gpu_busy_percent","memory_controller_busy_percent","device_used_mib","power_watts","temperature_c","sm_clock_mhz","memory_clock_mhz")
    return dict(samples=len(samples),values={name:dict(mean=statistics.mean(x[i] for x in samples),min=min(x[i] for x in samples),max=max(x[i] for x in samples)) for i,name in enumerate(names)})


def main():
    path = ROOT/"source/tools/benchmark_completion_api.py"
    if not path.exists():
        path = ROOT.parent.parent/"tools/benchmark_completion_api.py"
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "summarize"]
    namespace = dict(statistics=statistics, defaultdict=defaultdict)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"),namespace)
    result = dict(scope="One-repeat screening, not promotion or completed performance goal",achieved_hardware_counters=None,experiments=[])
    for label in ("", "recovery-v2", "unified-recovery-v3"):
        directory = ROOT/"v100"/label
        campaign = read(directory/"screen-report.json")
        for item in campaign["experiments"]:
            f = directory/item["label"]/"screen/report.json"
            if not f.exists():
                continue
            benchmark = read(f)
            if not benchmark["waves"]:
                continue
            rows = namespace["summarize"](benchmark["waves"])
            for row in rows:
                count = row["request_count"]
                waves = [w for w in benchmark["waves"] if w["request_count"] == count]
                assert all(w["emitted_tokens"] == 32*count for w in waves)
                row["all_measured_ids_equal_pinned_native"] = all(q["response"]["tokens"] == benchmark["case"]["expected_generated_ids"] for w in waves for q in w["requests"])
                row["telemetry"] = {str(g):telemetry(directory/item["label"], benchmark, count,g,campaign["utc_offset_seconds"]) for g in (1,0)}
            result["experiments"].append(dict(label=item["label"],campaign=label or "initial-v1",engine=item["engine"],mode=item["mode"],mtp_depth=item["mtp_depth"],whole_arm_completed=item["completed"],benchmark_completed=benchmark["completed"],summary=rows))
    result["pair_candidate"] = read(ROOT/"v100/pair-screen-v1/report.json")
    result["source_gguf"] = read(ROOT/"v100/export-report.json")
    (ROOT/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    for item in result["experiments"]:
        print(item["label"], "complete" if item["whole_arm_completed"] else "partial", [(r["request_count"],round(r["aggregate_complete_request_output_tps"],2)) for r in item["summary"]])


if __name__ == "__main__":
    main()
