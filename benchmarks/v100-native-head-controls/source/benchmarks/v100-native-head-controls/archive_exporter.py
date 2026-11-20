"""Export terminal metadata with shared compression; never read model binaries."""
import base64
import hashlib
import io
import json
import pathlib
import socket
import tarfile

assert socket.gethostname() == 'wzu-SYS-4029GP-TRT'
p = pathlib.Path('/home/data/wangyue/artifacts') / name
assert name.startswith('venturi-native-head-') and '/' not in name
assert (p / 'exit-code.txt').is_file()
manifest = {}
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for f in sorted(p.rglob('*')):
        if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts or f.suffix in ['.so', '.pt', '.sqlite', '.nsys-rep', '.pyc'] or f.name == 'core' or f.name.startswith('core.'):
            continue
        assert f.stat().st_size < 100 * 1024 * 1024
        data = f.read_bytes()
        rel = str(f.relative_to(p))
        h = hashlib.sha256(data).hexdigest()
        manifest[rel] = dict(sha256=h, bytes=len(data))
        if known.get(rel) == h:
            continue
        info = tarfile.TarInfo(rel)
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    data = json.dumps(dict(done=True, manifest=manifest, exit_code=int((p / 'exit-code.txt').read_text())), indent=2).encode()
    info = tarfile.TarInfo('__capture_manifest__.json')
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))
data = buffer.getvalue()
assert len(data) < 8 * 1024 * 1024, 'Finite metadata export bound; inspect unexpected volume'
h = hashlib.sha256(data).hexdigest()
cache = pathlib.Path('/home/data/wangyue/artifacts/venturi-head-metadata-exports-v1')
cache.mkdir(exist_ok=True)
packed = cache / (h + '.tar.gz')
if not packed.exists():
    packed.write_bytes(data)
assert hashlib.sha256(packed.read_bytes()).hexdigest() == h
parts = [data[i:i+8192] for i in range(0, len(data), 8192)]
result = dict(sha256=h, bytes=len(data), chunk_sha256=[hashlib.sha256(chunk).hexdigest() for chunk in parts])
if len(data) <= 16000:
    result['payload'] = base64.b64encode(data).decode()
print(json.dumps(result), flush=True)
