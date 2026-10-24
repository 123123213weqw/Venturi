import pathlib,json,hashlib,os,subprocess,time
root=pathlib.Path('/home/data/wangyue');base=root/'artifacts';names=['venturi-native-batch-v1','venturi-native-batch-v2','venturi-native-batch-v3','venturi-native-batch-v4','venturi-native-batch-final-v5','venturi-native-batch-diagnostics-v6'];states={}
for name in names:
 p=base/name;assert (p/'exit-code.txt').exists();meta=json.loads((p/'process.json').read_text());pid=meta['pid'];proc=pathlib.Path(f'/proc/{pid}');state=(proc/'stat').read_text().split(') ',1)[1].split()[0] if proc.exists() else 'gone';assert state in ['gone','Z'],(name,pid,state);states[name]=dict(pid=pid,state=state,exit_code=int((p/'exit-code.txt').read_text().strip()))
owned=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if any(s in cmd for s in ['tools/serve_native_benchmark.py','tools/qualify_qwen_batch.py','tools/profile_qwen_batch.py']):owned.append(dict(pid=int(p.name),cmd=cmd))
 except (OSError,UnicodeError):pass
assert not owned,owned
source=base/'venturi-native-batch-final-v5/executed-source.json';proof=json.loads(source.read_text());src=root/'repos/codex-build/Venturi/native-batch-src-v3'
for path,sha in proof['source_sha256'].items():assert hashlib.sha256((src/path).read_bytes()).hexdigest()==sha
result=dict(checked_unix=time.time(),workers=states,owned_inference_processes=owned,exact_executed_source_still_matches=True,executed_source_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),counter_policy=next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s))
print(json.dumps(result,indent=2))
