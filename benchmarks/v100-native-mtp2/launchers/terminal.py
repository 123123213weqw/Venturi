import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path('/home/data/wangyue');repo=r/'repos/codex-build/Venturi';workers={}
final=r/'artifacts/venturi-native-mtp2-full-profile-v14';deadline=time.monotonic()+120
while not (final/'exit-code.txt').exists():
 pid=json.loads((final/'process.json').read_text())['pid'];cmd=pathlib.Path(f'/proc/{pid}/cmdline');assert cmd.exists() and str(final/'worker.py').encode() in cmd.read_bytes(),'Final worker absent without terminal report';assert time.monotonic()<deadline,'Bounded terminal wait elapsed';time.sleep(5)
versions=['v1','model-v2','campaign-v3','final-v4','prefix-v5','prefix-model-v6','prefix-campaign-v7','prefix-final-v8','prefix-model-recovery-v9','prefix-campaign-v10','prefix-final-v11','prefix-tests-v12','prefix-tests-v13','full-profile-v14']
for n in versions:
 p=r/'artifacts'/('venturi-native-mtp2-'+n);pid=json.loads((p/'process.json').read_text())['pid'];s=pathlib.Path(f'/proc/{pid}/stat');state=s.read_text().split()[2] if s.exists() else None;assert state in [None,'Z'],(n,state)
 code=p/'exit-code.txt';rc=int(code.read_text()) if code.exists() else None
 if n in ['prefix-model-v6','prefix-final-v8']:assert rc==1 and (p/'failure.txt').is_file()
 elif n=='prefix-campaign-v7':assert rc is None and 'AssertionError' in (p/'worker.log').read_text()
 else:assert rc==0,(n,rc)
 workers[n]=dict(pid=pid,reported_exit_code=rc,remaining_process_state=state,terminal_observed=True,exit_reporting_limit='Uncaught predecessor assertion before wrapper terminal-file handling; worker traceback retained, OS exit status unavailable after reaping.' if rc is None else None)
owned=[];samplers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(p/'cwd')
  if 'nvidia-smi' in cmd and '-lms' in cmd:samplers.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
  if 'venturi-native-mtp2-' in cmd or cwd.startswith(str(repo/'native-mtp2-src-')):owned.append(dict(pid=int(p.name),cmdline=cmd,cwd=cwd))
 except OSError:pass
assert not owned,owned
ports=[s for s in subprocess.check_output(['ss','-ltnp'],text=True).splitlines() if ':18953 ' in s];assert not ports
snapshots={}
for snap in ['native-mtp2-src-v'+str(i) for i in [1,2,3,4,5,6]]:
 src=repo/snap;snapshots[snap]={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
assert '3651 MiB' in gpu and '3189 MiB' in gpu,gpu
print(json.dumps(dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,remaining_periodic_nvml_samplers=samplers,listening_ports=ports,executed_snapshots=snapshots,gpu_state=gpu,counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)),indent=2))
