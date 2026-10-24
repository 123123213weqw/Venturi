import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/home/data/wangyue');repo=r/'repos/codex-build/Venturi';workers={}
for name in ['study-v1','model-v2','campaign-v3','study-v4','model-v5','campaign-v6','final-v7']:
 p=r/'artifacts'/('venturi-token-order-'+name);pid=json.loads((p/'process.json').read_text())['pid'];s=pathlib.Path(f'/proc/{pid}/stat');state=s.read_text().split()[2] if s.exists() else None;assert state in [None,'Z'],(name,pid,state)
 rc=int((p/'exit-code.txt').read_text()) if (p/'exit-code.txt').exists() else None
 if name=='study-v1':assert rc==1 and '1 failed, 100 passed' in (p/'mtp-tests.log').read_text() and "AttributeError: 'ResidentLinear' object has no attribute 'vocab_size'" in (p/'mtp-tests.log').read_text()
 elif name=='model-v2':assert rc==1 and 'AssertionError' in (p/'failure.txt').read_text()
 elif name=='campaign-v3':assert rc is None and 'AssertionError' in (p/'worker.log').read_text() and not (p/'campaign-report.json').exists()
 else:assert rc==0,(name,rc)
 workers[name]=dict(pid=pid,exit_code=rc,remaining_process_state=state,preflight_only=(name in ['model-v2','campaign-v3']))
owned=[];samplers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
  if 'nvidia-smi' in cmd and '-lms' in cmd:samplers.append(dict(pid=int(p.name),cmdline=cmd))
  if 'venturi-token-order-' in cmd or cwd.startswith(str(repo/'token-order-src-')):owned.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
 except OSError:pass
assert not owned
ports=[s for s in subprocess.check_output(['ss','-ltnp'],text=True).splitlines() if ':18953 ' in s];assert not ports
snapshots={}
for snap in ['token-order-src-v1','token-order-src-v2']:
 src=repo/snap;snapshots[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True);assert '3651 MiB' in gpu and '3189 MiB' in gpu
print(json.dumps(dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,remaining_periodic_nvml_samplers=samplers,listening_ports=ports,executed_snapshots=snapshots,gpu_state=gpu,counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)),indent=2))
