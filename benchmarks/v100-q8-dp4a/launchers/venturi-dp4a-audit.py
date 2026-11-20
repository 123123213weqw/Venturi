import pathlib,json,hashlib,subprocess,time
root=pathlib.Path('/home/data/wangyue');base=root/'artifacts';names=['venturi-native-t8-screen-v1','venturi-dp4a-fixtures-v1','venturi-dp4a-projection-v2','venturi-tp-worker-screen-v1','venturi-llama-arithmetic-v1','venturi-dp4a-vector-v3','venturi-llama-arithmetic-v2','venturi-dp4a-final-v4'];workers={}
for name in names:
 p=base/name;assert (p/'exit-code.txt').exists();meta=json.loads((p/'process.json').read_text());pid=meta['pid'];stat=pathlib.Path(f'/proc/{pid}/stat');state=stat.read_text().split(') ',1)[1].split()[0] if stat.exists() else 'gone';assert state in ['gone','Z'];workers[name]=dict(pid=pid,state=state,exit_code=int((p/'exit-code.txt').read_text()))
owned=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if any(s in cmd for s in ['tools/serve_native_benchmark.py','benchmark_q8_dp4a.py','tools/profile_qwen_batch.py','--port 18946']):owned.append(dict(pid=int(p.name),cmd=cmd))
 except (OSError,UnicodeError):pass
assert not owned,owned
p=base/'venturi-dp4a-final-v4';src=root/'repos/codex-build/Venturi/dp4a-src-v4';proof=json.loads((p/'executed-source.json').read_text())
for rel,h in proof.items():assert hashlib.sha256((src/rel).read_bytes()).hexdigest()==h
result=dict(checked_unix=time.time(),workers=workers,owned_inference_processes=owned,executed_source_report_sha256=hashlib.sha256((p/'executed-source.json').read_bytes()).hexdigest(),gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),remaining_gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s))
print(json.dumps(result,indent=2))
