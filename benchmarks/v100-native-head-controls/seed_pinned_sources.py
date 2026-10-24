"""Seed byte-identical local source files, then require remote manifest validation.

No result, timing, response or numerical evidence is fabricated or synthesized.
Only sources whose hash matches the authenticated terminal observation are used.
Other seeded common metadata remains provisional until the remote capture check.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
terminal = json.loads((OUT / 'terminal-audit.json').read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
local = {sha(p): p for p in OUT.glob('*.py')}
verified = OUT / 'v100/controls-v7'
records = []
for name, state in terminal['workers'].items():
    if name.startswith('venturi-native-head-controls-profile-'):
        tag = 'profile-' + name.rsplit('-', 1)[1]
    elif name.startswith('venturi-native-head-controls-'):
        tag = 'controls-' + name.rsplit('-', 1)[1]
    else:
        tag = 'aligned-' + name.rsplit('-', 1)[1]
    directory = OUT / 'v100' / tag
    directory.mkdir(parents=True, exist_ok=True)
    worker = local.get(state['worker_sha256'])
    if worker:
        assert sha(worker) == state['worker_sha256']
        destination = directory / 'worker.py'
        if not destination.exists() or sha(destination) != sha(worker):
            shutil.copyfile(worker, destination)
            records.append(dict(tag=tag, file='worker.py', source=str(worker.relative_to(OUT)), sha256=sha(worker), authenticated_terminal_pin=True))
        tree = ast.parse(worker.read_text())
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                continue
            target = {'DIAGNOSTIC': 'aligned_teacher.py', 'PROFILE_SCRIPT': 'profile_ordinary256.py'}.get(statement.targets[0].id)
            if target and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                content = statement.value.value.encode()
                destination = directory / target
                if not destination.exists() or destination.read_bytes() != content:
                    destination.write_bytes(content)
                    records.append(dict(tag=tag, file=target, extracted_from_pinned_worker=True, sha256=hashlib.sha256(content).hexdigest()))
    for filename in ['executed-source.json', 'library.txt', 'counter-policy.txt']:
        source = verified / filename
        destination = directory / filename
        if source.is_file() and not destination.exists():
            shutil.copyfile(source, destination)
            records.append(dict(tag=tag, file=filename, source=str(source.relative_to(OUT)), sha256=sha(source), provisional_until_remote_manifest=True))
(OUT / 'source-deduplication-seeds.json').write_text(json.dumps(records, indent=2) + '\n')
print('Seeded', len(records), 'source/metadata records; all require subsequent remote manifest confirmation.')
