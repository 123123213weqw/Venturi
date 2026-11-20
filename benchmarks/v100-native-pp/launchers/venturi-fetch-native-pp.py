import pathlib,subprocess,json,tarfile,io
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi/benchmarks/v100-native-pp')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
meta='''import pathlib,json,hashlib,shutil,datetime
base=pathlib.Path('/home/wzu/codex-build/Venturi')
for name in ['v100-pp-final-v1','v100-pp-diagnostics-v3','v100-pp-comparison-v1','v100-pp-full-qualification-v1','v100-pp-validation-v1','v100-pp-validation-v2']:
 assert (base/name/'exit-code.txt').read_text().strip()=='0',name
p=base/'v100-pp-diagnostics-v3'
for mode in ['tp','pp']:
 for count in [1,8]:
  r=json.loads((p/f'{mode}-c{count}/profiled-run/report.json').read_text())
  assert r.get('completed') and r['profiled'] and len(r['concurrency'][0]['trials'])==1
assert '--cuda-event-trace=false' in (p/'worker.sh').read_text()
libs={}
for label,folder in [('initial','v100-pp-validation-v2'),('final','v100-pp-final-v1')]:
 lib=pathlib.Path((base/folder/'library.txt').read_text().strip());manifest=json.loads((lib.parent/'build.json').read_text());sha=hashlib.sha256(lib.read_bytes()).hexdigest();assert sha==manifest['library_sha256'];libs[label]=dict(path=str(lib),sha256=sha,manifest=manifest)
(p/'library-integrity.json').write_text(json.dumps(libs,indent=2)+'\\n')
now=datetime.datetime.now().astimezone();(p/'server-clock.json').write_text(json.dumps(dict(iso=now.isoformat(),utc_offset_seconds=int(now.utcoffset().total_seconds())),indent=2)+'\\n')
for job,source in [('v100-pp-validation-v1','v100-pp-src-v1'),('v100-pp-validation-v2','v100-pp-src-v2'),('v100-pp-comparison-v1','v100-pp-execution-src-v1')]:
 dst=base/job/'source-core';dst.mkdir(exist_ok=True);hashes={}
 for part in ['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/ops/csrc/qwen_pipeline.cuh','src/venturi/models/qwen_pipeline.py','src/venturi/models/qwen_native.py','tests/test_qwen_pipeline.py','src/venturi/ops/build.py']:
  f=base/source/part;out=dst/part;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out);hashes[part]=hashlib.sha256(out.read_bytes()).hexdigest()
 (base/job/'source-core-sha256.json').write_text(json.dumps(hashes,indent=2)+'\\n')
print('Terminal correctness/timing/diagnostic workers and library hashes verified')
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=meta,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
for folder,name in [('v100-pp-final-v1','final'),('v100-pp-diagnostics-v3','diagnostics'),('v100-pp-diagnostics-v2','failed-profiler-event-trace'),('v100-pp-comparison-v1','initial-comparison'),('v100-pp-full-qualification-v1','initial-qualification'),('v100-pp-validation-v1','validation-v1'),('v100-pp-validation-v2','validation-v2'),('v100-pp-diagnostics-v1','superseded-profiler-plan')]:
 cmd=f'tar -czf - --exclude="*.pt" --exclude="*.ninfer" --exclude="*.safetensors" --exclude="*.sqlite" --exclude="*.nsys-rep" --exclude="*.qdstrm" --exclude="*.so" --exclude="*.pyc" --exclude="__pycache__" -C /home/wzu/codex-build/Venturi/{folder} .'
 r=subprocess.run(['ssh']+opts+['WZU_Server',cmd],capture_output=True,timeout=120);r.check_returncode();dst=root/'v100'/name;dst.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tf:
  for m in tf.getmembers():
   p=pathlib.PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts
   if m.isdir():continue
   assert m.isfile();target=dst/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
 print(name,len(r.stdout),'compressed bytes')
