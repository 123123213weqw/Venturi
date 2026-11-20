import hashlib,json,pathlib,subprocess,time
base=pathlib.Path('/home/data/wangyue');repo=base/'repos/codex-build/Venturi';art=base/'artifacts'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
workers={}
for name in ['fixtures-v1','qualification-v2','campaign-v3','final-v4']:
 d=art/('venturi-shared-input-'+name);proc=read(d/'process.json');pid=proc['pid'];status=pathlib.Path(f'/proc/{pid}/stat');state=status.read_text().split(') ')[1].split()[0] if status.exists() else None
 workers[name]=dict(pid=pid,exit_code=int((d/'exit-code.txt').read_text()),remaining_process_state=state)
owned=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  raw=p.read_bytes().split(b'\0');args=[x.decode(errors='replace') for x in raw if x];exe=pathlib.Path(args[0]).name if args else ''
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if ('python' in exe or exe in ['nsys','llama-server']) and any('serve_native_benchmark.py' in a or 'profile_qwen_batch.py' in a or 'venturi-shared-input-' in a or a.endswith('llama-server') and 'v100-x1' in a for a in args):owned.append(dict(pid=int(p.parent.name),command=args))
proof={}
for name in ['shared-input-src-v1','shared-input-src-v2']:
 src=repo/name;proof[name]={str(p.relative_to(src)):sha(p) for folder in ['src','tests','tools'] for p in (src/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
source=read(art/'venturi-shared-input-fixtures-v1/build/77fedac214853798/build.json')['source_sha256'];assert all(sha(repo/name/'src/venturi/ops/csrc'/rel)==h for name in proof for rel,h in source.items())
result=dict(created_unix=time.time(),workers=workers,owned_inference_processes=owned,gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),counter_policy='\n'.join(l for l in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in l),listening_port_18948='\n'.join(l for l in subprocess.check_output(['ss','-ltn'],text=True).splitlines() if ':18948 ' in l),executed_snapshots=proof,native_source_sha256=source)
print(json.dumps(result,indent=2))
