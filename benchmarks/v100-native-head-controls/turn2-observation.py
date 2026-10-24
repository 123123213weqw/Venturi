from pathlib import Path
import subprocess,json,time
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
b=Path('benchmarks/v100-native-head-controls');code="""import json,pathlib,socket
p=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-head-controls-v1');r=dict(host=socket.gethostname(),exists=p.exists())
if p.exists():
 for f in ['process.json','exit-code.txt','failure.txt']:
  if (p/f).exists():r[f]=(p/f).read_text()
 if (p/'process.json').exists():
  pid=json.loads((p/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');r['live_cmdline']=q.read_bytes().decode(errors='replace') if q.exists() else None
print(json.dumps(r))
"""
r=subprocess.run(n['SSH'],input=code,text=True,capture_output=True,timeout=50)
d=dict(timestamp_unix=time.time(),rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
(b/'turn2-remote-observation.json').write_text(json.dumps(d,indent=2)+'\n');print(d,flush=True)
if not r.returncode:
 state=json.loads(r.stdout)
 if not state['exists']:
  q=subprocess.run(['../venturi-testenv/bin/python','/tmp/venturi-native-head-controls-v1-dispatch.py'],capture_output=True,text=True,timeout=90)
  (b/'dispatch-recovery-turn2.log').write_text(q.stdout+q.stderr);print(q.stdout,q.stderr,flush=True);q.check_returncode()
 else:print('Existing artifact found; inspect rather than relaunch.',flush=True)
else:
 q=subprocess.run(['tailscale','netcheck'],capture_output=True,text=True,timeout=35)
 (b/'tailscale-netcheck-turn2.log').write_text(q.stdout+q.stderr);print('NETCHECK',q.returncode,q.stdout,q.stderr,flush=True)
