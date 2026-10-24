import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/home/data/wangyue');repo=r/'repos/codex-build/Venturi';workers={}
final=r/'artifacts/venturi-t3-projection-final-v4';deadline=time.monotonic()+120
while not (final/'exit-code.txt').exists():
 pid=json.loads((final/'process.json').read_text())['pid'];p=pathlib.Path(f'/proc/{pid}/cmdline');assert p.exists() and str(final/'worker.py').encode() in p.read_bytes();assert time.monotonic()<deadline,'Bounded terminal wait elapsed';time.sleep(5)
for name in ['study-v1','model-v2','campaign-v3','final-v4']:
 p=r/'artifacts'/('venturi-t3-projection-'+name);pid=json.loads((p/'process.json').read_text())['pid'];s=pathlib.Path(f'/proc/{pid}/stat');state=s.read_text().split()[2] if s.exists() else None;assert state in [None,'Z'];rc=int((p/'exit-code.txt').read_text());assert rc==0;workers[name]=dict(pid=pid,exit_code=rc,remaining_process_state=state)
owned=[];samplers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
  if 'nvidia-smi' in cmd and '-lms' in cmd:samplers.append(dict(pid=int(p.name),cmdline=cmd))
  if 'venturi-t3-projection-' in cmd or cwd.startswith(str(repo/'t3-projection-src-')):owned.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
 except OSError:pass
assert not owned
ports=[s for s in subprocess.check_output(['ss','-ltnp'],text=True).splitlines() if ':18953 ' in s];assert not ports
snapshots={}
for snap in ['t3-projection-src-v1','t3-projection-src-v2']:
 src=repo/snap;snapshots[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True);assert '3651 MiB' in gpu and '3189 MiB' in gpu
print(json.dumps(dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,remaining_periodic_nvml_samplers=samplers,listening_ports=ports,executed_snapshots=snapshots,gpu_state=gpu,counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)),indent=2))
