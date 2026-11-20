import subprocess,os,signal,json,time
from pathlib import Path
C=Path(__file__).resolve().parent
script='''import hashlib,json,pathlib,subprocess
r=pathlib.Path('/home/data/wangyue/repos/codex-build/Venturi/five-src-v3');o=pathlib.Path('/home/data/wangyue/artifacts/venturi-five-window-v3')
a=json.loads((r/'local-source-pins.json').read_text());bad=[n for n,h in a.items() if not (r/n).is_file() or hashlib.sha256((r/n).read_bytes()).hexdigest()!=h]
assert not bad,bad
s=subprocess.check_output(['systemctl','--user','show','venturi-five-window-v3','-p','ActiveState','-p','SubState','-p','LoadState','-p','ExecMainStatus'],text=True)
if not o.exists() and 'ActiveState=active' not in s and 'SubState=running' not in s:
 subprocess.run(['systemd-run','--user','--unit=venturi-five-window-v3','--property=RuntimeMaxSec=40min','/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python','/home/data/wangyue/repos/codex-build/Venturi/five-control-v3/run-window-v3.py'],check=True)
 s=subprocess.check_output(['systemctl','--user','show','venturi-five-window-v3','-p','SubState','-p','ExecMainStatus'],text=True)
prod=subprocess.check_output(['docker','inspect','--format={{.State.Status}} {{.State.Health.Status}}','qwen38-27b'],text=True)
print(json.dumps(dict(files_verified=len(a),state=s,production=prod,files=[p.name for p in o.iterdir()] if o.exists() else [],exception=json.loads((o/'exception.json').read_text()) if (o/'exception.json').exists() else None)),flush=True)
'''
for attempt,(wait,route) in enumerate([(0,'fresh-ipv4'),(45,'fresh-ipv6'),(120,'existing-primary')]):
 if wait:time.sleep(wait)
 args=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2']
 if route=='fresh-ipv6':args+=['-6','-o','HostName=fd7a:115c:a1e0::5e35:bc73','-o','HostKeyAlias=[100.70.188.115]:9022']
 if route.startswith('fresh'):args+=['-o','ControlMaster=no','-o','ControlPath=none']
 t=time.time();p=subprocess.Popen(args+['WZU_Server','python3 -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
 try:o,e=p.communicate(script.encode(),timeout=30);rc=p.returncode
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL)
  for f in [p.stdin,p.stdout,p.stderr]:f.close()
  p.wait(timeout=5);o,e,rc=b'',b'outer timeout',124
 j=dict(attempt=attempt,route=route,rc=rc,stdout=o.decode(errors='replace'),stderr=e.decode(errors='replace'),seconds=time.time()-t,unix=time.time());(C/f'recover-dispatch-{attempt}.json').write_text(json.dumps(j,indent=2));print(json.dumps(j),flush=True)
 if rc==0 and 'files_verified' in j['stdout']:break
