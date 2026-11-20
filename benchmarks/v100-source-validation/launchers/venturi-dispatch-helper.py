from pathlib import Path
import subprocess
SSH=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
def dispatch(name,source):
 p=Path('/tmp/'+name+'-worker.py');p.write_text(source);compile(source,str(p),'exec')
 out='/home/data/wangyue/artifacts/'+name
 code=f'''import pathlib,subprocess,json,time
out=pathlib.Path({out!r});out.mkdir();p=out/'worker.py';p.write_text({source!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
 Path('/tmp/'+name+'-dispatch.py').write_text('import subprocess\ncode='+repr(code)+'\nr=subprocess.run('+repr(SSH)+',input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()\n')
 r=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=60);print(name,r.stdout,r.stderr);r.check_returncode()
