import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/home/data/wangyue');repo=r/'repos/codex-build/Venturi';workers={}
for name in ['venturi-shared-activation-v1','venturi-shared-activation-model-v2','venturi-shared-activation-campaign-v3','venturi-shared-activation-final-v4']:
 p=r/'artifacts'/name;pid=json.loads((p/'process.json').read_text())['pid'];s=pathlib.Path(f'/proc/{pid}/stat');state=s.read_text().split()[2] if s.exists() else None;assert state in [None,'Z'];rc=int((p/'exit-code.txt').read_text());assert rc==0;workers[name]=dict(pid=pid,exit_code=rc,remaining_process_state=state)
owned=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
  if 'venturi-shared-activation-' in cmd or cwd.startswith(str(repo/'shared-activation-src-')):owned.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
 except OSError:pass
assert not owned
port=[s for s in subprocess.check_output(['ss','-ltnp'],text=True).splitlines() if ':18952 ' in s];assert not port
snapshots={}
for snap in ['shared-activation-src-v1','shared-activation-src-v2']:
 src=repo/snap;snapshots[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
print(json.dumps(dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,listening_ports=port,executed_snapshots=snapshots,gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)),indent=2))
