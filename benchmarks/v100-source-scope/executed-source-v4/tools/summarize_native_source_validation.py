"""Count checkpoint metadata checks in explicitly marked native-call intervals.

strace durations are instrumented, not an estimate of untraced critical-path cost.
Unfinished/resumed records are paired by TID before interval filtering.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def summarize(trace, report):
    paths = set(report["checkpoint_paths"])
    intervals = report["calls"]
    pending = {}
    counts = Counter()
    durations = Counter()
    per_path = {p: Counter() for p in sorted(paths)}
    failures = []
    total = 0
    pattern = re.compile(r"^\s*(?:\[pid\s+)?(\d+)\]?\s+(\d+\.\d+)\s+(.*)$")
    for line in trace.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        tid, timestamp, call = match.groups()
        if "<unfinished ...>" in call:
            pending[tid] = (float(timestamp), call.split("<unfinished ...>")[0])
            continue
        if call.startswith("<...") and " resumed>" in call:
            if tid not in pending:
                continue
            timestamp, prefix = pending.pop(tid)
            call = prefix + call.split(" resumed>", 1)[1]
        timestamp = float(timestamp)
        if not any(c["start_unix"] <= timestamp <= c["end_unix"] for c in intervals):
            continue
        name = call.split("(", 1)[0]
        if name not in ("fstat", "stat", "newfstatat", "statx"):
            continue
        total += 1
        found = [p for p in paths if "<" + p + ">" in call or json.dumps(p) in call]
        if not found:
            continue
        assert len(found) == 1, "Ambiguous checkpoint annotation"
        path = found[0]
        if name == "fstat" or re.search(r"\d+<[^>]+>, \"\"", call):
            kind = "descriptor"
        elif name == "stat" or "AT_FDCWD" in call:
            kind = "pathname"
        else:
            kind = "unclassified"
        counts[kind] += 1
        per_path[path][kind] += 1
        duration = re.search(r"<([0-9.]+)>$", call)
        if duration:
            durations[kind] += float(duration.group(1))
        else:
            failures.append("Missing duration: " + line)
        if not re.search(r"= 0\s+<[0-9.]+>$", call):
            failures.append("Unsuccessful metadata call: " + line)
    return dict(
        scope="Instrumented syscall counts, not unprofiled throughput or critical-path attribution",
        checkpoint_calls=dict(counts),
        checkpoint_calls_by_path={p: dict(c) for p, c in per_path.items()},
        instrumented_syscall_seconds=dict(durations),
        all_traced_metadata_calls_in_intervals=total,
        failures=failures,
        unmatched_pending_tids=sorted(pending),
        raw_native_calls=len(intervals),
        model_outputs_exact=all(c["output_ids_equal"] for c in intervals),
        engine_change=not report.get("source_guards_unchanged", True),
        source_check_cadence_changed=not report.get("source_guards_unchanged", True),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("report")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = summarize(
        Path(a.trace).read_text(), json.loads(Path(a.report).read_text())
    )
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
