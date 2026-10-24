import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/home/data/wangyue');repo=r/'repos/codex-build/Venturi';workers={}
for name,expected in [('venturi-gdn-compact-v1',0),('venturi-gdn-compact-model-v2',0),('venturi-gdn-compact-campaign-v3',0),('venturi-gdn-compact-final-v4',1),('venturi-gdn-compact-final-v5',0)]:
 p=r/'artifacts'/name;pid=json.loads((p/'process.json').read_text())['pid'];s=pathlib.Path(f'/proc/{pid}/stat');state=s.read_text().split()[2] if s.exists() else None;assert state in [None,'Z'],(name,state);rc=int((p/'exit-code.txt').read_text());assert rc==expected;workers[name]=dict(pid=pid,exit_code=rc,remaining_process_state=state)
owned=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
  if 'venturi-gdn-compact-' in cmd or cwd.startswith(str(repo/'gdn-compact-src-')):owned.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
 except OSError:pass
assert not owned,owned
ports=[s for s in subprocess.check_output(['ss','-ltnp'],text=True).splitlines() if ':18953 ' in s];assert not ports
snapshots={}
for snap in ['gdn-compact-src-v1','gdn-compact-src-v2']:
 src=repo/snap;snapshots[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
assert '3651 MiB' in gpu and '3189 MiB' in gpu,gpu
print(json.dumps(dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,listening_ports=ports,executed_snapshots=snapshots,gpu_state=gpu,counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)),indent=2))
