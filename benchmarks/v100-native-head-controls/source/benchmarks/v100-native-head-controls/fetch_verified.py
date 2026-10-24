"""Resume small hash-verified evidence records after interrupted SSH streams.

Only terminal, explicitly named benchmark artifacts are readable. Large tensor,
binary and profiler files remain remote; their recorded hashes are retained.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zlib

OUT = Path(__file__).resolve().parent
JOBS = {
    "controls-v1": "venturi-native-head-controls-v1",
    "aligned-v2": "venturi-native-head-aligned-v2",
    "controls-v4": "venturi-native-head-controls-v4",
    "aligned-v5": "venturi-native-head-aligned-v5",
    "controls-v7": "venturi-native-head-controls-v7",
    "aligned-v8": "venturi-native-head-aligned-v8",
    "profile-v9": "venturi-native-head-controls-profile-v9",
}
SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ForwardAgent=no",
    "-o", "StrictHostKeyChecking=yes", "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/venturi-v100-head-recover-%C",
    "-o", "ControlPersist=600", "-o", "ProxyCommand=none",
    "-o", "ConnectTimeout=45", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4", "WZU_Server", "python3 -",
]
REMOTE = r'''
import pathlib,hashlib,json,zlib,base64,socket
assert socket.gethostname()=='wzu-SYS-4029GP-TRT'
p=pathlib.Path('/home/data/wangyue/artifacts')/name
assert (p/'exit-code.txt').is_file(),'Artifact not terminal; avoid compression during timing'
manifest={}
for f in sorted(p.rglob('*')):
 if not f.is_file() or f.is_symlink():continue
 rel=str(f.relative_to(p))
 if '__pycache__' in f.parts or f.suffix in ['.so','.pt','.sqlite','.nsys-rep','.pyc'] or f.name=='core' or f.name.startswith('core.'):continue
 assert f.stat().st_size<100*1024*1024,'Unexpected large evidence file; inspect rather than transfer'
 data=f.read_bytes();h=hashlib.sha256(data).hexdigest();manifest[rel]=dict(sha256=h,bytes=len(data))
 if known.get(rel)==h:continue
 print(json.dumps(dict(path=rel,sha256=h,bytes=len(data),payload=base64.b64encode(zlib.compress(data,9)).decode())),flush=True)
print(json.dumps(dict(done=True,manifest=manifest,exit_code=int((p/'exit-code.txt').read_text()))),flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(directory, rel):
    candidate = Path(rel)
    assert not candidate.is_absolute() and ".." not in candidate.parts
    assert candidate.parts and ".env" not in candidate.parts
    return directory / candidate


def consume(stdout, directory):
    manifest, transferred = None, 0
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            # An interrupted final line is not a verified file.
            continue
        if item.get("done"):
            manifest = item
            continue
        p = safe_path(directory, item["path"])
        data = zlib.decompress(base64.b64decode(item["payload"], validate=True))
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_name(p.name + ".verified-transfer-tmp")
        temp.write_bytes(data)
        temp.replace(p)
        transferred += 1
    if manifest:
        assert all(safe_path(directory, rel).is_file() and sha(safe_path(directory, rel)) == v["sha256"] for rel, v in manifest["manifest"].items())
    return manifest, transferred


def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", choices=sorted(JOBS))
    args = parser.parse_args()
    for tag in args.tags:
        directory = OUT / "v100" / tag
        directory.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            if attempt:
                time.sleep(10)
            known = {str(p.relative_to(directory)): sha(p) for p in directory.rglob("*") if p.is_file()}
            wire = "name=" + repr(JOBS[tag]) + "\nknown=" + repr(known) + "\n" + REMOTE
            start = time.monotonic()
            try:
                r = subprocess.run(SSH, input=wire, text=True, capture_output=True, timeout=240)
                stdout, stderr, rc = r.stdout, r.stderr, r.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
                rc = None
            manifest, transferred = consume(stdout, directory)
            record = dict(tag=tag, attempt=attempt, elapsed_seconds=time.monotonic()-start, rc=rc, stderr=stderr, verified_transferred=transferred, terminal_manifest_received=manifest is not None)
            with (OUT / "verified-fetch.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if manifest:
                (OUT / (tag + "-capture-manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
                break
        else:
            raise RuntimeError(f"Evidence capture unfinished for {tag}; verified partial records retained")


if __name__ == "__main__":
    main()
