"""Fetch shared-compressed terminal evidence in bounded hash-verified chunks."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import tarfile
import time
import zlib

OUT = Path(__file__).resolve().parent
BASE = runpy.run_path(str(OUT / 'fetch_verified_batched.py'))
sha = BASE['BASE']['sha']
safe = BASE['BASE']['safe_path']
SSH = BASE['SSH']
CACHE = Path('/tmp/venturi-v100-metadata-chunks')
CACHE.mkdir(exist_ok=True)

def call(wire, label):
    for attempt in range(2):
        if attempt:
            time.sleep(5)
        start = time.monotonic()
        try:
            result = subprocess.run(SSH, input=wire, text=True, capture_output=True, timeout=120)
            data = json.loads(result.stdout) if result.returncode == 0 else None
            record = dict(label=label, attempt=attempt, rc=result.returncode, stderr=result.stderr, response_bytes=len(result.stdout), elapsed_seconds=time.monotonic()-start)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            data = None
            record = dict(label=label, attempt=attempt, error=type(exc).__name__, elapsed_seconds=time.monotonic()-start)
        with (OUT / 'archive-fetch.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if data is not None:
            return data
    raise RuntimeError('Finite archive transport attempts exhausted: ' + label)

def consume(data, directory):
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        member = archive.getmember('__capture_manifest__.json')
        manifest = json.loads(archive.extractfile(member).read())
        for item in archive.getmembers():
            if item.name == '__capture_manifest__.json':
                continue
            assert item.isfile() and item.name in manifest['manifest']
            p = safe(directory, item.name)
            content = archive.extractfile(item).read()
            assert hashlib.sha256(content).hexdigest() == manifest['manifest'][item.name]['sha256']
            assert len(content) == manifest['manifest'][item.name]['bytes']
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + f'.archive-tmp-{os.getpid()}')
            tmp.write_bytes(content)
            tmp.replace(p)
    assert all(safe(directory, rel).is_file() and sha(safe(directory, rel)) == row['sha256'] for rel, row in manifest['manifest'].items())
    return manifest

def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs='+', choices=sorted(BASE['JOBS']))
    args = parser.parse_args()
    for tag in args.tags:
        directory = OUT / 'v100' / tag
        directory.mkdir(parents=True, exist_ok=True)
        capture = OUT / (tag + '-capture-manifest.json')
        if capture.exists():
            m = json.loads(capture.read_text())
            assert all(sha(directory / rel) == row['sha256'] for rel, row in m['manifest'].items())
            print(tag, 'already fully verified', flush=True)
            continue
        known = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
        packed = base64.b64encode(zlib.compress(json.dumps(known).encode(), 9)).decode()
        wire = 'import base64,json,zlib\nname=' + repr(BASE['JOBS'][tag]) + '\nknown=json.loads(zlib.decompress(base64.b64decode(' + repr(packed) + ')))\n' + (OUT / 'archive_exporter.py').read_text()
        result = call(wire, tag + '-prepare')
        h = result['sha256']
        assert len(h) == 64 and all(c in '0123456789abcdef' for c in h)
        if 'payload' in result:
            data = base64.b64decode(result['payload'], validate=True)
        else:
            parts = []
            for index, expected in enumerate(result['chunk_sha256']):
                p = CACHE / (h + '-' + str(index))
                if not p.exists() or sha(p) != expected:
                    command = "import pathlib,json,base64,hashlib\np=pathlib.Path('/home/data/wangyue/artifacts/venturi-head-metadata-exports-v1')/" + repr(h + '.tar.gz') + '\nwith p.open(\'rb\') as f:\n f.seek(' + str(index*8192) + ');data=f.read(8192)\nprint(json.dumps(dict(payload=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())),flush=True)\n'
                    chunk = call(command, tag + '-chunk-' + str(index))
                    content = base64.b64decode(chunk['payload'], validate=True)
                    assert hashlib.sha256(content).hexdigest() == chunk['sha256'] == expected
                    p.write_bytes(content)
                parts.append(p.read_bytes())
            data = b''.join(parts)
        assert len(data) == result['bytes'] and hashlib.sha256(data).hexdigest() == h
        manifest = consume(data, directory)
        manifest['transport'] = dict(shared_compression=True, archive_sha256=h, compressed_bytes=len(data), chunk_bytes=8192)
        capture.write_text(json.dumps(manifest, indent=2) + '\n')
        print(tag, 'verified', len(manifest['manifest']), 'files;', len(data), 'compressed bytes', flush=True)

if __name__ == '__main__':
    main()
