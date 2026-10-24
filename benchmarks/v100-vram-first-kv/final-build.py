import json,subprocess,time
from pathlib import Path
W=Path('/home/data/wangyue/artifacts/venturi-kv-tier-window-v1')
for _ in range(900):
 if (W/'production-restored.json').exists():
  r=json.loads((W/'production-restored.json').read_text());assert r['healthy'] and r['config_unchanged'];break
 time.sleep(2)
else:raise RuntimeError('First window restoration prerequisite missing; no overlapping GPU work')
subprocess.run(['/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python',str(Path(__file__).parent/'fixture-worker-v3.py')],check=True,timeout=540)
