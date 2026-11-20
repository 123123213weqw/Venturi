from pathlib import Path
import subprocess,json,hashlib,shutil,shlex,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-mtp-draft-residual'
code=r'''import pathlib,json,subprocess,time,hashlib
root=pathlib.Path('/home/data/wangyue');last=root/'artifacts/venturi-mtp-draft-residual-guards-v5';deadline=time.monotonic()+1200
while not (last/'exit-code.txt').exists():
 pid=json.loads((last/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(last/'worker.py').encode() in q.read_bytes(),'Last worker actually missing without terminal state'
 if time.monotonic()>deadline:raise TimeoutError('Last worker verified live; observation window elapsed, do not restart')
 time.sleep(5)
workers={}
for label in ['fixtures-v1','model-v2','campaign-v3','profile-v4','guards-v5']:
 p=root/('artifacts/venturi-mtp-draft-residual-'+label);workers[label]=dict(process=json.loads((p/'process.json').read_text()),exit_code=int((p/'exit-code.txt').read_text()))
ps=subprocess.run(['ps','-eo','pid,ppid,pgid,args'],text=True,capture_output=True,check=True).stdout
owned=[l for l in ps.splitlines() if 'venturi-mtp-draft-residual' in l and '<defunct>' not in l]
samplers=[l for l in ps.splitlines() if 'nvidia-smi' in l and '-lms' in l]
ports=subprocess.run(['ss','-ltnp'],capture_output=True,text=True,check=True).stdout
source=root/'repos/codex-build/Venturi/mtp-draft-residual-src-v1'
manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (source/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
payloads={}
for p in [root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer',root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer']:
 s=p.stat();payloads[str(p)]=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
print(json.dumps(dict(ended_unix=time.time(),workers=workers,owned_processes=owned,remaining_periodic_samplers=samplers,port18953=[l for l in ports.splitlines() if ':18953' in l],gpu_after=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout,counter_policy=pathlib.Path('/proc/driver/nvidia/params').read_text(),source_manifest=manifest,payload_identity=payloads),indent=2))
'''
p=subprocess.run(ns['SSH'],input=code,text=True,capture_output=True,timeout=1270)
(out/'terminal-audit-observation.log').write_text(p.stderr)
p.check_returncode();r=json.loads(p.stdout);(out/'terminal-audit.json').write_text(json.dumps(r,indent=2)+'\n')
(out/'v100').mkdir(exist_ok=True)
for label in ['fixtures-v1','model-v2','campaign-v3','profile-v4','guards-v5']:
 dest=out/'v100'/label;dest.mkdir(exist_ok=True);cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
 for pattern in ['*.so','*.sqlite','*.nsys-rep','*.pt','__pycache__/']:cmd+=['--exclude',pattern]
 cmd+=['WZU_Server:/home/data/wangyue/artifacts/venturi-mtp-draft-residual-'+label+'/',str(dest)+'/']
 for attempt in range(2):
  q=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
  if not q.returncode:break
  if attempt==1:q.check_returncode()
  time.sleep(3)
 print(label,r['workers'][label]['exit_code'],flush=True)
assert all(v['exit_code']==0 for v in r['workers'].values()),'Terminal failure preserved; inspect it without blind restart'
assert not r['owned_processes'] and not r['remaining_periodic_samplers'] and not r['port18953']
assert '3651 MiB' in r['gpu_after'] and '3189 MiB' in r['gpu_after']
m=json.loads((out/'source-manifest-v1.json').read_text());assert m==r['source_manifest']
for rel,h in m.items():
 p=root/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,rel
 dst=out/'executed-source-v1'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
pin=root/'benchmarks/v100-native-column-mlp/fresh-payload-and-external-bundle-pins.json';old=json.loads(pin.read_text())
for p,identity in r['payload_identity'].items():
 previous=old['checkpoint_payloads'][p];assert all(identity[k]==previous[k] for k in identity)
(out/'reused-full-payload-pins.json').write_text(json.dumps(dict(prior_full_hash_report=str(pin),prior_full_hash_report_sha256=hashlib.sha256(pin.read_bytes()).hexdigest(),stable_target_and_draft_identity=True,identity=r['payload_identity'],full_payload_hashes={p:old['checkpoint_payloads'][p]['sha256'] for p in r['payload_identity']},new_full_target_hash_scan=False,scope='Reuse previously full-hashed payloads with unchanged dev/inode/size/mtime/ctime; current trained draft qualifier additionally rehashes its full payload. No fresh external engine run.'),indent=2)+'\n')
print('All five owned workers terminal; source snapshot and unchanged payload identities preserved; original GPU footprint restored')
