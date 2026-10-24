"""One-shot dependency worker: wait for verified source, then run CPU Q8 export.

Never starts/restarts a download or stops a source/GPU service. A source-wait
expiry stops only this observer; it is not evidence that the source job stopped.
This is a single build-pipeline dependency, not a recurring scheduler.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep, time


def write_state(path, record):
    temporary = path.with_suffix(".next")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def source_alive(pid, source):
    proc = Path(f"/proc/{pid}/cmdline")
    if not proc.exists():
        return False
    command = proc.read_bytes().split(b"\0")
    aliases = {str(source.absolute()).encode(), str(source.resolve()).encode()}
    return any(b"download_qwen_source.py" in arg for arg in command) and bool(
        aliases.intersection(command)
    )


def run(args):
    source, output, report, job = (
        Path(p).expanduser().absolute()
        for p in (args.source, args.output, args.report, args.job_directory)
    )
    job.mkdir(parents=True, exist_ok=True)
    with (job / ".lock").open("a+") as lock:
        # A duplicate observer cannot publish a second converter or replace the
        # existing worker's state. This lock is held through child completion.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            output.exists()
            or output.is_symlink()
            or report.exists()
            or report.is_symlink()
        ):
            raise FileExistsError(
                "Existing output/report requires inspection, never replacement"
            )
        state = dict(
            status="waiting-source",
            watcher_pid=os.getpid(),
            source_pid=args.source_pid,
            source=str(source),
            output=str(output),
            report=str(report),
            started_unix=time(),
        )
        state_path = job / "state.json"
        deadline = monotonic() + args.source_wait_seconds
        while True:
            record = json.loads((source / "venturi-source.json").read_text())
            state.update(
                source_status=record.get("status"),
                source_revision=record.get("revision"),
                checked_unix=time(),
            )
            if record.get("status") == "complete":
                break
            if record.get("status") == "failed":
                state["status"] = "source-failed"
                write_state(state_path, state)
                return 4
            alive = source_alive(args.source_pid, source)
            state["source_process_matches"] = alive
            if not alive:
                state["status"] = "source-stopped-before-complete"
                write_state(state_path, state)
                return 5
            if monotonic() >= deadline:
                state["status"] = "source-wait-expired-observer-only"
                write_state(state_path, state)
                return 6
            write_state(state_path, state)
            sleep(args.poll_seconds)
        # The converter independently rejects partial sources, verifies every
        # recorded file hash and protects publication against races/mutations.
        command = [
            sys.executable,
            str(Path(__file__).with_name("export_qwen_q8.py")),
            str(source),
            str(output),
            "--report",
            str(report),
        ]
        child = subprocess.Popen(command, env=os.environ.copy())
        state.update(
            status="export-running", export_pid=child.pid, export_started_unix=time()
        )
        write_state(state_path, state)
        code = child.wait()  # Observation never kills a progressing export.
        state.update(export_returncode=code, finished_unix=time())
        if code == 0:
            result = json.loads(report.read_text())
            if (
                Path(result["output"]) != output
                or output.stat().st_size != result["file_bytes"]
            ):
                raise RuntimeError("Successful converter report/output mismatch")
            state.update(
                status="complete",
                artifact_id=result["artifact_id"],
                artifact_sha256=result["sha256"],
            )
        else:
            state["status"] = "export-failed"
        write_state(state_path, state)
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "report", "job-directory"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source-pid", type=int, required=True)
    parser.add_argument("--source-wait-seconds", type=float, default=21600)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if (
        args.source_pid <= 0
        or not 0 <= args.source_wait_seconds <= 86400
        or not 1 <= args.poll_seconds <= 300
    ):
        raise ValueError("Invalid source PID/wait/poll limits")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
