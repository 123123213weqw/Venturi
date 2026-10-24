"""Bounded exact-container restoration, independently callable by a watchdog."""
import hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/home/data/wangyue/artifacts/venturi-kv-tier-window-v1')
def fingerprint(a):
 return hashlib.sha256(json.dumps(dict(args=a['Args'],env=a['Config']['Env'],mounts=sorted(a['Mounts'],key=lambda v:(v.get('Destination',''),v.get('Source',''),v.get('Type','')))),sort_keys=True).encode()).hexdigest()
def inspect():return json.loads(subprocess.check_output(['docker','inspect','qwen38-27b'],timeout=15))[0]
def save(n,j):
 p=ROOT/n;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def restore(watchdog=False):
 if watchdog:
  subprocess.run(['systemctl','--user','stop','venturi-kv-tier-window-v1'],timeout=50,check=False)
 before=json.loads((ROOT/'production-before.json').read_text());a=inspect()
 assert a['Id']==before['container_id'],'Container replaced externally; do not touch unrelated instance'
 assert fingerprint(a)==before['canonical_config_fingerprint'],'Container configuration changed externally'
 if not a['State']['Running']:subprocess.run(['docker','start',a['Id']],check=True,timeout=90)
 result=dict(container_id=a['Id'],canonical_config_fingerprint=fingerprint(a),config_unchanged=True,watchdog=watchdog)
 for _ in range(90):
  try:
   a=inspect();h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4))
   if a['State']['Running'] and a['State'].get('Health',{}).get('Status')=='healthy' and h.get('status')=='ok':
    result.update(healthy=True,health_api=h,started_at=a['State']['StartedAt'],ended_unix=time.time())
    assert a['Id']==before['container_id'] and fingerprint(a)==before['canonical_config_fingerprint']
    result['models_api']=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5))
    result['counter_policy']=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s]
    assert result['counter_policy']==before['counter_policy']
    save('production-restored-watchdog.json' if watchdog else 'production-restored.json',result);return result
  except (OSError,ValueError,subprocess.SubprocessError):pass
  time.sleep(2)
 result.update(healthy=False,ended_unix=time.time());save('restore-failed.json',result);raise RuntimeError('Production restoration health check failed')
if __name__=='__main__':
 try:print(json.dumps(restore('--watchdog' in sys.argv)),flush=True)
 except BaseException as e:save('restore-exception.json',dict(error=repr(e),unix=time.time()));raise
