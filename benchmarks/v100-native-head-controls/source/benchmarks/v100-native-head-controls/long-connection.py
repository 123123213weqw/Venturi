from pathlib import Path
import subprocess,json,time
b=Path('benchmarks/v100-native-head-controls');n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);ssh=n['SSH'][:]
for old,new in [('ControlMaster=no','ControlMaster=auto'),('ControlPath=none','ControlPath=/tmp/venturi-v100-head-controls-%C'),('ConnectTimeout=15','ConnectTimeout=90'),('ServerAliveInterval=10','ServerAliveInterval=30'),('ServerAliveCountMax=2','ServerAliveCountMax=6')]:ssh[ssh.index(old)]=new
ssh[-2:-2]=['-o','ControlPersist=600'];(b/'long-connection-route.json').write_text(json.dumps(dict(ssh=ssh,reason='Gateway/self-address changed; cached peer RTT19s exceeds the earlier15s handshake timeout. Own reusable control session, no global settings changed.'),indent=2)+'\n')
wire=(b/'delta-v4-v5-observe-and-dispatch.py').read_text();print('One guarded authenticated connection;90s handshake/300s overall; owns temporary reusable session',flush=True)
try:r=subprocess.run(ssh,input=wire,text=True,capture_output=True,timeout=300)
except subprocess.TimeoutExpired as e:(b/'long-connection-dispatch.log').write_text(str(e));print('Bounded observation timeout; inspect handles through the same session before any next action',flush=True);raise SystemExit(0)
result=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr,ended_unix=time.time());(b/'long-connection-dispatch.log').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if not r.returncode:(b/'long-connection-success.json').write_text(json.dumps(result,indent=2)+'\n')
