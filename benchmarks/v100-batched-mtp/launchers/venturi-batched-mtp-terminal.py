import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/home/data/wangyue');repo=r/'repos/codex-build/Venturi';workers={}
for name in ['fixtures-v1','model-v2','campaign-v3','final-v4','sdk-tests-v5','sdk-campaign-v6','sdk-tests-v7','sdk-campaign-v8','sdk-final-v9']:
 p=r/'artifacts'/('venturi-batched-mtp-'+name);pid=json.loads((p/'process.json').read_text())['pid'];s=pathlib.Path(f'/proc/{pid}/stat');state=s.read_text().split()[2] if s.exists() else None
 assert state in [None,'Z'],(name,pid,state)
 rc=int((p/'exit-code.txt').read_text()) if (p/'exit-code.txt').exists() else None
 if name=='sdk-tests-v5':assert rc==1 and '2 failed, 1990 passed, 448 skipped' in (p/'full-tests.log').read_text()
 elif name=='sdk-campaign-v6':assert rc is None and 'AssertionError' in (p/'worker.log').read_text() and not (p/'campaign-report.json').exists()
 else:assert rc==0,(name,rc)
 workers[name]=dict(pid=pid,exit_code=rc,remaining_process_state=state,preflight_only_no_speed_run=(name=='sdk-campaign-v6'))
owned=[];samplers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
  if 'nvidia-smi' in cmd and '-lms' in cmd:samplers.append(dict(pid=int(p.name),cmdline=cmd))
  if 'venturi-batched-mtp-' in cmd or cwd.startswith(str(repo/'batched-mtp-src-')):owned.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
 except OSError:pass
assert not owned
ports=[s for s in subprocess.check_output(['ss','-ltnp'],text=True).splitlines() if ':18953 ' in s];assert not ports
snapshots={};launcher_files={}
for snap in ['batched-mtp-src-v1','batched-mtp-src-v2','batched-mtp-src-v3','batched-mtp-src-v4','batched-mtp-src-v5']:
 src=repo/snap;snapshots[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for snap in snapshots:
 src=repo/snap;launcher_files[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src/'scripts').rglob('*') if p.is_file()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True);assert '3651 MiB' in gpu and '3189 MiB' in gpu
print(json.dumps(dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,remaining_periodic_nvml_samplers=samplers,listening_ports=ports,executed_snapshots=snapshots,launcher_files=launcher_files,gpu_state=gpu,counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)),indent=2))
