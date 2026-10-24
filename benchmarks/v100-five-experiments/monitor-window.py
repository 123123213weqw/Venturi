import subprocess,os,signal,json,time
from pathlib import Path
C=Path(__file__).resolve().parent
script='''import pathlib,json,subprocess
b=pathlib.Path('/home/data/wangyue/artifacts/venturi-five-v2');r=pathlib.Path('/home/data/wangyue/artifacts/venturi-five-window-v4')
files=sorted(p.name for p in r.iterdir()) if r.exists() else []
ex={n:json.loads((r/n).read_text()) for n in ['exception.json','production-paused.json','production-restored.json','completed.json'] if (r/n).exists()}
print(json.dumps(dict(build_completed=(b/'completed.json').exists(),build_fixtures=(b/'fixtures.log').read_text()[-250:] if (b/'fixtures.log').exists() else None,state=subprocess.check_output(['systemctl','--user','show','venturi-five-build-v2','venturi-five-window-v4','-p','SubState','-p','ExecMainStatus'],text=True),files=files,markers=ex)),flush=True)
'''
last=None
for i in range(18):
 if i:time.sleep(120)
 cmd=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=8','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);t=time.time()
 try:o,e=p.communicate(script.encode(),timeout=30);rc=p.returncode
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL)
  for f in [p.stdin,p.stdout,p.stderr]:f.close()
  p.wait(timeout=5);o,e,rc=b'',b'outer timeout',124
 j=dict(rc=rc,stdout=o.decode(errors='replace'),stderr=e.decode(errors='replace'),unix=time.time(),seconds=time.time()-t);(C/f'monitor-{i:02}.json').write_text(json.dumps(j,indent=2))
 if j['stdout']!=last or rc:print(json.dumps(j),flush=True);last=j['stdout']
 if rc==0:
  try:
   data=json.loads(j['stdout']);m=data['markers']
   if 'production-restored.json' in m or ('exception.json' in m and 'production-paused.json' not in m):break
  except ValueError:pass
