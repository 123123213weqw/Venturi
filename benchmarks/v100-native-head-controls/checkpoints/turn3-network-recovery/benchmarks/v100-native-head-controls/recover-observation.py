from pathlib import Path
import subprocess,json,time
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
b=Path('benchmarks/v100-native-head-controls');code="""import pathlib,json,subprocess
p=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-head-controls-v1')
r=dict(exists=p.exists())
if p.exists():
 for f in ['process.json','exit-code.txt','failure.txt']:
  if (p/f).exists():r[f]=(p/f).read_text()
 if (p/'process.json').exists():
  pid=json.loads((p/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');r['cmdline']=q.read_bytes().decode(errors='replace') if q.exists() else None
print(json.dumps(r))
"""
for attempt in range(3):
 if attempt:time.sleep([0,5,15][attempt])
 r=subprocess.run(n['SSH'],input=code,text=True,capture_output=True,timeout=60)
 with (b/'network-recovery.log').open('a') as f:f.write(json.dumps(dict(attempt=attempt,rc=r.returncode,stdout=r.stdout,stderr=r.stderr))+'\n')
 print(r.stdout,r.stderr,flush=True)
 if r.returncode:continue
 d=json.loads(r.stdout)
 if not d['exists']:
  print('Verified no artifact/worker; exact serialized dispatch recovery',flush=True)
  q=subprocess.run(['../venturi-testenv/bin/python','/tmp/venturi-native-head-controls-v1-dispatch.py'],capture_output=True,text=True,timeout=90)
  (b/'dispatch-recovery.log').write_text(q.stdout+q.stderr);print(q.stdout,q.stderr,flush=True)
  q.check_returncode()
 else:print('Existing outcome; do not dispatch again',flush=True)
 break
else:print('Three bounded observations failed; unfinished network-dependent dispatch, not a proven hard block',flush=True)
