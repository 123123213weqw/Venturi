import subprocess,os,signal,json,hashlib,time,tarfile
from pathlib import Path
C=Path(__file__).resolve().parent
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=8','-o','ServerAliveCountMax=2','WZU_Server']
def run(command,file,limit):
 with file.open('wb') as f:
  p=subprocess.Popen(ssh+[command],stdout=f,stderr=subprocess.PIPE,start_new_session=True);t=time.time()
  try:_,err=p.communicate(timeout=limit);rc=p.returncode
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGKILL);p.stderr.close();p.wait(timeout=5);err=b'outer timeout';rc=124
 return dict(rc=rc,stderr=err.decode(errors='replace'),seconds=time.time()-t,bytes=file.stat().st_size)
pin=C/'archive-pin.json';j=run('cat /home/data/wangyue/artifacts/venturi-five-evidence-v1.json',pin,25);assert j['rc']==0,j
record=json.loads(pin.read_text());archive=C/'evidence.tgz';j=run('cat '+record['path'],archive,120)
j.update(expected_bytes=record['bytes'],actual_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),expected_sha256=record['sha256'])
j['verified']=j['bytes']==record['bytes'] and j['actual_sha256']==record['sha256'];(C/'archive-fetch.json').write_text(json.dumps(j,indent=2));print(json.dumps(j),flush=True);assert j['verified'],j
out=C/'evidence'
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  assert member.isfile(),member.name
  target=(out/member.name).resolve();assert target.is_relative_to(out.resolve())
  assert not target.exists(),target
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
manifest=json.loads((out/'manifest.json').read_text())
bad=[n for n,pin in manifest['included'].items() if (out/n).stat().st_size!=pin['bytes'] or hashlib.sha256((out/n).read_bytes()).hexdigest()!=pin['sha256']]
assert not bad,bad
(C/'archive-members-verified.json').write_text(json.dumps(dict(verified=True,files=len(manifest['included']),all_sha256_match=True),indent=2)+'\n');print('verified raw members',len(manifest['included']))
