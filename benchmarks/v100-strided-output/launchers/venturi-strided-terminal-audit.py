import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/home/data/wangyue');a=r/'artifacts';repo=r/'repos/codex-build/Venturi'
names=['fixtures-v1','campaign-v2','fixtures-v3','campaign-v4','campaign-v5','final-v6','external-depth-v7','final-v8','external-depth-v9'];workers={}
for n in names:
 p=a/('venturi-strided-'+n);record=json.loads((p/'process.json').read_text());pid=record['pid'];proc=pathlib.Path(f'/proc/{pid}');state=(proc/'stat').read_text().split()[2] if (proc/'stat').exists() else None
 assert state in [None,'Z'],(n,pid,state)
 workers[n]=dict(pid=pid,exit_code=int((p/'exit-code.txt').read_text()) if (p/'exit-code.txt').exists() else None,remaining_process_state=state)
expected={'fixtures-v1':1,'fixtures-v3':0,'campaign-v5':0,'final-v6':1,'external-depth-v7':1,'final-v8':0,'external-depth-v9':0}
assert all(workers[k]['exit_code']==v for k,v in expected.items());assert workers['campaign-v2']['exit_code'] is None and workers['campaign-v4']['exit_code'] is None
owned=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
  if 'venturi-strided-' in cmd or cwd.startswith(str(repo/'strided-src-')):owned.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
 except (OSError,PermissionError):pass
assert not owned,owned
ports=subprocess.check_output(['ss','-ltnp'],text=True);listening=[s for s in ports.splitlines() if ':18950 ' in s or ':18951 ' in s];assert not listening
snapshots={}
for snap in ['strided-src-v1','strided-src-v2']:
 src=repo/snap;snapshots[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
print(json.dumps(dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,listening_ports=listening,gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s),executed_snapshots=snapshots),indent=2))
