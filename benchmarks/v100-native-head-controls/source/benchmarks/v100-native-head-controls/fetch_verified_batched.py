"""Bound evidence responses to small resumable batches on the unstable route."""

import argparse
import base64
import json
from pathlib import Path
import runpy
import subprocess
import time
import zlib

OUT = Path(__file__).resolve().parent
BASE = runpy.run_path(str(OUT / "fetch_verified.py"))
JOBS = dict(BASE["JOBS"])
JOBS.update({
    "controls-v10": "venturi-native-head-controls-v10",
    "aligned-v11": "venturi-native-head-aligned-v11",
    "profile-v12": "venturi-native-head-controls-profile-v12",
    "controls-v13": "venturi-native-head-controls-v13",
    "aligned-v14": "venturi-native-head-aligned-v14",
    "profile-v15": "venturi-native-head-controls-profile-v15",
    "aligned-v16": "venturi-native-head-aligned-v16",
    "controls-v17": "venturi-native-head-controls-v17",
    "aligned-v18": "venturi-native-head-aligned-v18",
})
SSH = BASE["SSH"][:]
for value in ["ControlPath=/tmp/venturi-v100-head-recover-%C", "ControlPersist=600"]:
    index = SSH.index(value)
    del SSH[index-1:index+1]
SSH[SSH.index("ControlMaster=auto")] = "ControlMaster=no"
REMOTE = BASE["REMOTE"].replace(
    "manifest={}\n", "manifest={};wire_bytes=0;more=False\n"
).replace(
    " print(json.dumps(dict(path=rel,sha256=h,bytes=len(data),payload=base64.b64encode(zlib.compress(data,9)).decode())),flush=True)",
    " line=json.dumps(dict(path=rel,sha256=h,bytes=len(data),payload=base64.b64encode(zlib.compress(data,9)).decode()))\n"
    " if wire_bytes and wire_bytes+len(line)>24000:more=True;break\n"
    " print(line,flush=True);wire_bytes+=len(line)"
).replace(
    "print(json.dumps(dict(done=True,manifest=manifest,exit_code=int((p/'exit-code.txt').read_text()))),flush=True)",
    "if more:print(json.dumps(dict(batch_more=True,wire_bytes=wire_bytes)),flush=True)\n"
    "else:print(json.dumps(dict(done_zlib=base64.b64encode(zlib.compress(json.dumps(dict(done=True,manifest=manifest,exit_code=int((p/'exit-code.txt').read_text()))).encode(),9)).decode())),flush=True)"
)
assert "batch_more=True" in REMOTE and "done_zlib=" in REMOTE
REMOTE = REMOTE.replace("if more:print", "if not more and wire_bytes:more=True\nif more:print")


def consume(stdout, directory):
    lines, more = [], False
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("batch_more"):
            more = True
        elif item.get("done_zlib"):
            lines.append(zlib.decompress(base64.b64decode(item["done_zlib"], validate=True)).decode())
        else:
            lines.append(line)
    manifest, transferred = BASE["consume"]("\n".join(lines), directory)
    return manifest, transferred, more


def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", choices=sorted(JOBS))
    args = parser.parse_args()
    for tag in args.tags:
        directory = OUT / "v100" / tag
        directory.mkdir(parents=True, exist_ok=True)
        capture = OUT / (tag + "-capture-manifest.json")
        if capture.exists():
            assert all(BASE["sha"](directory / rel) == row["sha256"] for rel, row in json.loads(capture.read_text())["manifest"].items())
            print(tag + ": previous full capture reverified; no transfer repeated", flush=True)
            continue
        stalled = 0
        for batch in range(64):
            known = {str(p.relative_to(directory)): BASE["sha"](p) for p in directory.rglob("*") if p.is_file()}
            packed = base64.b64encode(zlib.compress(json.dumps(known).encode(), 9)).decode()
            wire = "import json,zlib,base64\nname=" + repr(JOBS[tag]) + "\nknown=json.loads(zlib.decompress(base64.b64decode(" + repr(packed) + ")))\n" + REMOTE
            start = time.monotonic()
            try:
                result = subprocess.run(SSH, input=wire, text=True, capture_output=True, timeout=120)
                stdout, stderr, rc = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
                rc = None
            manifest, transferred, more = consume(stdout, directory)
            record = dict(tag=tag, batch=batch, response_bytes=len(stdout.encode()), elapsed_seconds=time.monotonic()-start, rc=rc, stderr=stderr, verified_transferred=transferred, more_marker=more, terminal_manifest_received=manifest is not None)
            with (OUT / "verified-fetch-batched.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if manifest:
                capture.write_text(json.dumps(manifest, indent=2) + "\n")
                break
            stalled = 0 if transferred else stalled + 1
            if stalled >= 2:
                raise RuntimeError(f"Two batches without progress for {tag}; verified partial files preserved")
            if rc or not more:
                time.sleep(10)
        else:
            raise RuntimeError(f"Finite batch bound reached for {tag}; evidence capture unfinished")


if __name__ == "__main__":
    main()
