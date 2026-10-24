from pathlib import Path
import subprocess,json,shlex,time,hashlib
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);out=Path('benchmarks/v100-native-head-gdn');ssh=n['SSH'].copy();ssh[ssh.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6'
code=r'''import pathlib,json,time,subprocess,hashlib
root=pathlib.Path('/home/data/wangyue');last=root/'artifacts/venturi-native-head-gdn-profile-v6';deadline=time.monotonic()+600
while not (last/'exit-code.txt').exists():
 pid=json.loads((last/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(last/'worker.py').encode() in q.read_bytes(),'Last worker missing without terminal marker'
 if time.monotonic()>deadline:raise TimeoutError('Verified live profile still pending; observation timeout is not terminal')
 time.sleep(5)
workers={}
for tag in ['fixtures-v1','fixtures-v2','model-v3','mtp-v4','campaign-v5','profile-v6']:
 p=root/('artifacts/venturi-native-head-gdn-'+tag);workers[tag]=dict(process=json.loads((p/'process.json').read_text()),exit_code=int((p/'exit-code.txt').read_text()))
ps=subprocess.run(['ps','-eo','pid,ppid,pgid,args'],capture_output=True,text=True,check=True).stdout;ports=subprocess.run(['ss','-ltnp'],capture_output=True,text=True,check=True).stdout
manifests={}
for version in (1,2,3):
 src=root/('repos/codex-build/Venturi/native-head-gdn-src-v'+str(version));manifests[str(version)]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
payloads={}
for p in [root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer',root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer']:
 s=p.stat();payloads[str(p)]=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
print(json.dumps(dict(ended_unix=time.time(),workers=workers,owned_processes=[x for x in ps.splitlines() if 'venturi-native-head-gdn-' in x and '<defunct>' not in x],remaining_periodic_samplers=[x for x in ps.splitlines() if 'nvidia-smi' in x and '-lms' in x],port18953=[x for x in ports.splitlines() if ':18953' in x],gpu_after=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout,counter_policy=pathlib.Path('/proc/driver/nvidia/params').read_text(),source_manifests=manifests,payload_identity=payloads),indent=2))
'''
for attempt in range(3):
 if attempt:time.sleep(5*attempt)
 r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=690)
 with (out/'terminal-observation.log').open('a') as f:f.write(str(attempt)+'\n'+r.stderr)
 if not r.returncode:break
 if attempt==2:r.check_returncode()
audit=json.loads(r.stdout);(out/'terminal-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print({k:v['exit_code'] for k,v in audit['workers'].items()},flush=True)
for tag in ['mtp-v4','campaign-v5','profile-v6']:
 dest=out/'v100'/tag;dest.mkdir(parents=True,exist_ok=True);cmd=['rsync','-az','--timeout=90','-e',shlex.join(ssh[:-2])]
 for pattern in ['*.so','*.pt','*.sqlite','*.nsys-rep','__pycache__/']:cmd+=['--exclude',pattern]
 cmd+=['WZU_Server:/home/data/wangyue/artifacts/venturi-native-head-gdn-'+tag+'/',str(dest)+'/']
 for attempt in range(2):
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
  with (out/'fetch.log').open('a') as f:f.write(tag+' '+str(attempt)+'\n'+p.stdout+p.stderr)
  if not p.returncode:break
  if attempt==1:p.check_returncode()
  time.sleep(5)
 print('fetched',tag,flush=True)
for version in (1,2,3):
 expected=json.loads((out/f'source-manifest-v{version}.json').read_text());assert expected==audit['source_manifests'][str(version)];assert all(hashlib.sha256((out/f'executed-source-v{version}'/rel).read_bytes()).hexdigest()==h for rel,h in expected.items())
assert {k:v['exit_code'] for k,v in audit['workers'].items()}=={'fixtures-v1':1,'fixtures-v2':1,'model-v3':1,'mtp-v4':0,'campaign-v5':0,'profile-v6':0},'Preserved failure; inspect before recovery'
assert not audit['owned_processes'] and not audit['remaining_periodic_samplers'] and not audit['port18953']
assert '3651 MiB' in audit['gpu_after'] and '3189 MiB' in audit['gpu_after'] and 'RmProfilingAdminOnly: 1' in audit['counter_policy']
prior=Path('benchmarks/v100-gdn-head-pilot/reused-full-payload-pins.json');pins=json.loads(prior.read_text());assert pins['identity']==audit['payload_identity']
(out/'reused-full-payload-pins.json').write_text(json.dumps(dict(prior_reference=str(prior),prior_reference_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),identity=audit['payload_identity'],full_payload_hashes=pins['full_payload_hashes'],new_full_target_hash_scan=False,draft_full_hash_fresh_in_mtp_qualification=True),indent=2)+'\n')
print('Six workers terminal; sources and original payload identities verified; GPU baseline restored',flush=True)
