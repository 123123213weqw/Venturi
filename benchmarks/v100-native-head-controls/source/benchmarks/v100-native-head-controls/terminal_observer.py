"""Read terminal state and provenance; never start or terminate benchmark jobs."""
import base64
import hashlib
import json
import pathlib
import socket
import subprocess
import time
import zlib

assert socket.gethostname() == 'wzu-SYS-4029GP-TRT'
root = pathlib.Path('/home/data/wangyue')
artifacts = root / 'artifacts'
names = ['venturi-native-head-controls-v1', 'venturi-native-head-aligned-v2', 'venturi-native-head-controls-v4', 'venturi-native-head-aligned-v5', 'venturi-native-head-controls-v7', 'venturi-native-head-aligned-v8', 'venturi-native-head-controls-profile-v9', 'venturi-native-head-controls-v10', 'venturi-native-head-aligned-v11', 'venturi-native-head-controls-profile-v12', 'venturi-native-head-controls-v13', 'venturi-native-head-aligned-v14', 'venturi-native-head-controls-profile-v15', 'venturi-native-head-aligned-v16', 'venturi-native-head-controls-v17', 'venturi-native-head-aligned-v18']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

workers = {}
for name in names:
    p = artifacts / name
    process = json.loads((p / 'process.json').read_text())
    command = pathlib.Path('/proc') / str(process['pid']) / 'cmdline'
    workers[name] = dict(process=process, exit_code=int((p / 'exit-code.txt').read_text()), worker_sha256=sha(p / 'worker.py'), current_cmdline=command.read_bytes().decode(errors='replace') if command.exists() else None)
    assert not workers[name]['current_cmdline'] or str(p / 'worker.py') not in workers[name]['current_cmdline'], 'Worker still live; inspect rather than assuming completion'
src = root / 'repos/codex-build/Venturi/native-head-gdn-src-v3'
source = {str(p.relative_to(src)): sha(p) for folder in ['src', 'tests', 'tools'] for p in (src / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
library = pathlib.Path((artifacts / 'venturi-native-head-controls-v17/library.txt').read_text().strip())
paths = [root / 'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer', root / 'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer', root / 'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf', pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin/llama-server')]
payloads = {}
for p in paths:
    s = p.stat()
    payloads[str(p)] = dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
processes = subprocess.run(['ps', '-eo', 'pid,ppid,pgid,args'], capture_output=True, text=True, check=True).stdout
samplers = []
for row in processes.splitlines():
    if 'nvidia-smi' not in row or '-lms' not in row:
        continue
    pid = row.split()[0]
    fd = pathlib.Path('/proc') / pid / 'fd/1'
    try:
        destination = str(fd.readlink())
    except OSError:
        destination = None
    samplers.append(dict(process=row, stdout_destination=destination, owned=bool(destination and '/artifacts/venturi-native-head-' in destination)))
ports = subprocess.run(['ss', '-ltnp'], capture_output=True, text=True, check=True).stdout
result = dict(observed_unix=time.time(), hostname=socket.gethostname(), workers=workers, source_manifest=source, library_sha256=sha(library), payload_identity=payloads, remaining_artifact_processes=[row for row in processes.splitlines() if '/artifacts/venturi-native-head-' in row and '<defunct>' not in row], periodic_samplers=samplers, port18953=[row for row in ports.splitlines() if ':18953' in row], gpu_after=subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout, counter_policy=pathlib.Path('/proc/driver/nvidia/params').read_text())
data = json.dumps(result, indent=2).encode()
print(json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), zlib_base64=base64.b64encode(zlib.compress(data, 9)).decode())), flush=True)
