import subprocess,pathlib,tarfile,io,json
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi/benchmarks/v100-native-tp')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
meta='''import pathlib,json,hashlib,shutil,datetime,subprocess
base=pathlib.Path('/home/wzu/codex-build/Venturi');p=base/'v100-tp-diagnostics-v3'
for name in ['v100-tp-full-v1','v100-tp-diagnostics-v3','v100-tp-validation-v2','v100-warp-experiment-v1']:
 assert (base/name/'exit-code.txt').read_text().strip()=='0',name
library=pathlib.Path((base/'v100-tp-validation-v2/library.txt').read_text().strip())
manifest=json.loads((library.parent/'build.json').read_text());sha=hashlib.sha256(library.read_bytes()).hexdigest();assert sha==manifest['library_sha256']
(p/'library-integrity.json').write_text(json.dumps(dict(path=str(library),sha256=sha,manifest=manifest),indent=2)+'\\n')
now=datetime.datetime.now().astimezone();(p/'server-clock.json').write_text(json.dumps(dict(iso=now.isoformat(),utc_offset_seconds=int(now.utcoffset().total_seconds())),indent=2)+'\\n')
for job,source in [('v100-tp-validation-v1','v100-tp-src-v1'),('v100-warp-experiment-v1','v100-warp-src-v1')]:
 dst=base/job/'source-core';dst.mkdir(exist_ok=True)
 names=['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/models/qwen_native.py','src/venturi/ops/build.py','tests/test_q8_tiling.py','tools/benchmark_q8_tiles.py']
 hashes={}
 for name in names:
  f=base/source/name
  if not f.exists():continue
  out=dst/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out);hashes[name]=hashlib.sha256(out.read_bytes()).hexdigest()
 (base/job/'source-core-sha256.json').write_text(json.dumps(hashes,indent=2)+'\\n')
warp=base/'v100-warp-experiment-v1'
for entry in json.loads((warp/'libraries.json').read_text()):
 lib=pathlib.Path(entry['library']);assert hashlib.sha256(lib.read_bytes()).hexdigest()==entry['manifest']['library_sha256']
 shutil.copy2(lib.parent/'build.log',warp/f"build-{entry['name']}.log")
container='fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d'
state=subprocess.check_output(['docker','inspect','--format','name={{.Name}} status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}',container],text=True).strip()
cmd=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .Config.Cmd}}',container],text=True))
allowed={'-m','--model','--split-mode','--tensor-split','--spec-type','--spec-draft-n-max'}
selected={x:cmd[i+1] for i,x in enumerate(cmd[:-1]) if x in allowed}
(p/'service-state.json').write_text(json.dumps(dict(state=state,selected_flags=selected,scope='Only requested service state and benchmark-relevant flags; no environment or authentication flags'),indent=2)+'\\n')
print('Terminal success and native library integrity verified')
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=meta,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
for folder,name in [('v100-tp-full-v1','full'),('v100-tp-diagnostics-v3','diagnostics'),('v100-tp-diagnostics-v2','diagnostics-failed-incomplete-deploy'),('v100-tp-diagnostics-v1','diagnostics-failed-collection'),('v100-tp-validation-v2','validation-v2'),('v100-tp-validation-v1','validation-v1-failed'),('v100-warp-experiment-v1','warp')]:
 cmd=f'tar -czf - --exclude="*.pt" --exclude="*.ninfer" --exclude="*.safetensors" --exclude="*.sqlite" --exclude="*.nsys-rep" --exclude="*.qdstrm" --exclude="*.so" --exclude="*.pyc" --exclude="python-deps" --exclude="__pycache__" -C /home/wzu/codex-build/Venturi/{folder} .'
 r=subprocess.run(['ssh']+opts+['WZU_Server',cmd],capture_output=True,timeout=120);r.check_returncode()
 dst=root/'v100'/name;dst.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tf:
  for m in tf.getmembers():
   p=pathlib.PurePosixPath(m.name)
   assert not p.is_absolute() and '..' not in p.parts
   if m.isdir():continue
   assert m.isfile()
   target=dst/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
 print(name,len(r.stdout),'compressed bytes')
(root/'v100/server-clock.json').write_bytes((root/'v100/diagnostics/server-clock.json').read_bytes())
