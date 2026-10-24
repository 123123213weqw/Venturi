from pathlib import Path
import subprocess,json,shlex,time
b=Path('benchmarks/v100-native-head-controls')
inner=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=60','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-W','%h:%p','WZU_4080_Tail']
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=auto','-o','ControlPath=/tmp/venturi-v100-head-via4080-%C','-o','ControlPersist=600','-o','ConnectTimeout=90','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-o','ProxyCommand='+shlex.join(inner),'WZU_Server','python3 -']
(b/'via4080-original-port-route.json').write_text(json.dumps(dict(ssh=ssh,scope='Existing authorized4080 jump reaches the same selected V100 original9022 service; no benchmark executes on the4080. Host keys/agent boundaries unchanged.'),indent=2)+'\n')
try:r=subprocess.run(ssh,input=(b/'delta-v4-v5-observe-and-dispatch.py').read_text(),text=True,capture_output=True,timeout=300)
except subprocess.TimeoutExpired as e:(b/'via4080-original-port.log').write_text(str(e));print('Bounded same-host jump observation/dispatch timeout; inspect before replay',flush=True);raise SystemExit(0)
result=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr,ended_unix=time.time());(b/'via4080-original-port.log').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if not r.returncode:(b/'via4080-success.json').write_text(json.dumps(result,indent=2)+'\n')
