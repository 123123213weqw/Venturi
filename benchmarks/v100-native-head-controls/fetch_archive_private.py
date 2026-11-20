"""Isolate evidence export from stalled shared SSH data streams."""
import hashlib
import json
from pathlib import Path
import runpy

OUT = Path(__file__).resolve().parent
assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
module = runpy.run_path(str(OUT / 'fetch_archive.py'))
ssh = module['SSH'][:]
ssh[ssh.index('ControlMaster=no')] = 'ControlMaster=auto'
ssh[ssh.index('ConnectTimeout=45')] = 'ConnectTimeout=90'
ssh[-2:-2] = ['-o', 'ControlPath=/tmp/venturi-v100-head-export-%C', '-o', 'ControlPersist=600', '-o', 'Compression=yes']
module['call'].__globals__['SSH'] = ssh
original = module['call']

def isolated_call(wire, label):
    return original(wire, 'isolated-export:' + label)

module['main'].__globals__['call'] = isolated_call
(OUT / 'private-archive-route.json').write_text(json.dumps(dict(ssh=ssh, exporter_sha256=hashlib.sha256((OUT / 'fetch_archive.py').read_bytes()).hexdigest(), reason='Small shared-master queries recovered, but later bulk sessions stalled. Isolate read-only metadata export on an owned bounded connection; do not alter or terminate the user control master.'), indent=2) + '\n')
if __name__ == '__main__':
    module['main']()
