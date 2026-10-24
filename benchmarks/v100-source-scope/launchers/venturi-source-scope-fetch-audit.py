from pathlib import Path
import subprocess,json,shlex,time,hashlib
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);out=Path('benchmarks/v100-source-scope')
code=r'''import pathlib,json,time,subprocess,hashlib
root=pathlib.Path('/home/data/wangyue');last=root/'artifacts/venturi-source-scope-profile-v5';deadline=time.monotonic()+1800
while not (last/'exit-code.txt').exists():
 pid=json.loads((last/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(last/'worker.py').encode() in q.read_bytes(),'Last worker missing without terminal file'
 if time.monotonic()>deadline:raise TimeoutError('Verified last worker still live; observation window elapsed, do not restart')
 time.sleep(5)
workers={}
for label in ['fixtures-v1','fixtures-v2','model-v3','campaign-v4','profile-v5']:
 p=root/('artifacts/venturi-source-scope-'+label);workers[label]=dict(process=json.loads((p/'process.json').read_text()),exit_code=int((p/'exit-code.txt').read_text()))
ps=subprocess.run(['ps','-eo','pid,ppid,pgid,args'],capture_output=True,text=True,check=True).stdout
owned=[l for l in ps.splitlines() if 'venturi-source-scope-' in l and '<defunct>' not in l]
samplers=[l for l in ps.splitlines() if 'nvidia-smi' in l and '-lms' in l]
ports=subprocess.run(['ss','-ltnp'],capture_output=True,text=True,check=True).stdout
manifests={}
for n in range(1,5):
 src=root/('repos/codex-build/Venturi/source-scope-src-v'+str(n));manifests[str(n)]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
payloads={}
for p in [root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer',root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer']:
 s=p.stat();payloads[str(p)]=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
print(json.dumps(dict(ended_unix=time.time(),workers=workers,owned_processes=owned,remaining_periodic_samplers=samplers,port18953=[l for l in ports.splitlines() if ':18953' in l],gpu_after=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout,counter_policy=pathlib.Path('/proc/driver/nvidia/params').read_text(),source_manifests=manifests,payload_identity=payloads),indent=2))
'''
r=subprocess.run(ns['SSH'],input=code,text=True,capture_output=True,timeout=1880);(out/'terminal-audit-observation.log').write_text(r.stderr);r.check_returncode();audit=json.loads(r.stdout);(out/'terminal-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
def sync(remote,dest,excludes=()):
 dest.mkdir(parents=True,exist_ok=True);cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
 for pattern in excludes:cmd+=['--exclude',pattern]
 cmd+=['WZU_Server:'+remote,str(dest)+'/']
 for attempt in range(2):
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
  if not p.returncode:return
  if attempt==1:p.check_returncode()
  time.sleep(3)
for label in audit['workers']:
 sync('/home/data/wangyue/artifacts/venturi-source-scope-'+label+'/',out/'v100'/label,['*.so','*.sqlite','*.nsys-rep','*.pt','__pycache__/']);print(label,audit['workers'][label]['exit_code'],flush=True)
for n in range(1,5):
 expected=json.loads((out/('source-manifest-v'+str(n)+'.json')).read_text());assert expected==audit['source_manifests'][str(n)]
 dst=out/('executed-source-v'+str(n))
 for directory in ['src','tests','tools']:
  sync('/home/data/wangyue/repos/codex-build/Venturi/source-scope-src-v'+str(n)+'/'+directory+'/',dst/directory,['__pycache__/','*.pyc'])
 assert all(hashlib.sha256((dst/rel).read_bytes()).hexdigest()==h for rel,h in expected.items())
assert {k:v['exit_code'] for k,v in audit['workers'].items()}=={'fixtures-v1':1,'fixtures-v2':0,'model-v3':0,'campaign-v4':0,'profile-v5':0},'Terminal failure preserved; inspect before any recovery'
assert not audit['owned_processes'] and not audit['remaining_periodic_samplers'] and not audit['port18953']
assert '3651 MiB' in audit['gpu_after'] and '3189 MiB' in audit['gpu_after']
for rel,h in audit['source_manifests']['4'].items():assert hashlib.sha256(Path(rel).read_bytes()).hexdigest()==h,rel
prior=json.loads(Path('benchmarks/v100-mtp-draft-residual/reused-full-payload-pins.json').read_text());assert prior['identity']==audit['payload_identity']
(out/'reused-full-payload-pins.json').write_text(json.dumps(dict(prior_full_hash_reference='benchmarks/v100-mtp-draft-residual/reused-full-payload-pins.json',prior_reference_sha256=hashlib.sha256(Path('benchmarks/v100-mtp-draft-residual/reused-full-payload-pins.json').read_bytes()).hexdigest(),identity=audit['payload_identity'],full_payload_hashes=prior['full_payload_hashes'],new_full_target_hash_scan=False,fresh_draft_hash_in_qualification=True,scope='Prior full target/draft hashes reused with unchanged identity; new trained execution qualifier additionally rehashes the full draft payload. No fresh external campaign.'),indent=2)+'\n')
print('Five workers terminal including preserved initial fixture issue; four executed sources verified; payload identities stable; original GPU footprint restored')
