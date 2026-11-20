import hashlib,json,os,signal,subprocess,tarfile,time
from pathlib import Path
C=Path(__file__).parent
BASE=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=8','-o','ServerAliveCountMax=2','-o','IPQoS=none','-o','ControlMaster=no','-o','ControlPath=none']
def call(args,timeout):
 p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
 try:o,e=p.communicate(timeout=timeout);return p.returncode,o,e
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL)
  for f in [p.stdout,p.stderr]:f.close()
  p.wait(timeout=5);return 124,b'',b'bounded outer timeout'
r,o,e=call(BASE+['WZU_Server','cat /home/data/wangyue/artifacts/venturi-kv-tier-evidence-v1.pin.json'],35);assert r==0,e
pin=json.loads(o);(C/'archive-pin.json').write_text(json.dumps(pin,indent=2)+'\n')
routes=[[],['-6','-o','HostName=fd7a:115c:a1e0::5e35:bc73','-o','HostKeyAlias=[100.70.188.115]:9022','-o','ControlMaster=auto','-o','ControlPersist=300','-o','ControlPath=/tmp/venturi-kv-ipv6-%r-%h-%p']]
attempts=[]
for options in routes:
 started=time.time();r,o,e=call(BASE+options+['WZU_Server','cat '+pin['path']],180)
 got=hashlib.sha256(o).hexdigest();ok=r==0 and len(o)==pin['bytes'] and got==pin['sha256']
 attempts.append(dict(route='ipv6' if options else 'primary',rc=r,seconds=time.time()-started,bytes=len(o),sha256=got,verified=ok,stderr=e.decode(errors='replace')[-1000:]))
 (C/'archive-transfer.json').write_text(json.dumps(attempts,indent=2)+'\n')
 if ok:(C/'evidence.tgz').write_bytes(o);break
 time.sleep(3)
else:raise RuntimeError('Archive not recovered; preserve remote evidence, try chunked verified transfer next')
out=C/'evidence';out.mkdir(exist_ok=False)
with tarfile.open(C/'evidence.tgz') as tf:
 for m in tf.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 tf.extractall(out,filter='data')
m=json.loads((out/'manifest.json').read_text())
for n,pin in m.items():
 p=out/n;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'],n
(C/'archive-members-verified.json').write_text(json.dumps(dict(members=len(m),all_equal=True),indent=2)+'\n');print('Verified',len(m),'raw evidence files')
