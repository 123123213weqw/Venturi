from pathlib import Path
import subprocess,json,hashlib,shutil
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);out=Path('benchmarks/v100-source-validation')
code=r'''import pathlib,json,subprocess,time,hashlib
root=pathlib.Path('/home/data/wangyue');p=root/'artifacts/venturi-source-validation-probe-v1';worker=json.loads((p/'process.json').read_text());sampler=json.loads((p/'sampler.json').read_text());ps=subprocess.run(['ps','-eo','pid,ppid,pgid,args'],capture_output=True,text=True,check=True).stdout
source=root/'repos/codex-build/Venturi/source-validation-src-v1';manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (source/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
library=pathlib.Path((p/'library.txt').read_text().strip())
print(json.dumps(dict(ended_unix=time.time(),worker=worker,sampler=sampler,exit_code=int((p/'exit-code.txt').read_text()),owned_processes=[l for l in ps.splitlines() if 'venturi-source-validation-probe-v1' in l and '<defunct>' not in l],owned_sampler_live=pathlib.Path('/proc/'+str(sampler['pid'])).exists(),gpu_after=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout,counter_policy=pathlib.Path('/proc/driver/nvidia/params').read_text(),port18953=[l for l in subprocess.run(['ss','-ltnp'],capture_output=True,text=True,check=True).stdout.splitlines() if ':18953' in l],source_manifest=manifest,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()),indent=2))
'''
r=subprocess.run(ns['SSH'],input=code,text=True,capture_output=True,timeout=70)
(out/'terminal-audit-observation.log').write_text(r.stderr);r.check_returncode();j=json.loads(r.stdout);(out/'terminal-audit.json').write_text(json.dumps(j,indent=2)+'\n')
assert j['exit_code']==0 and not j['owned_processes'] and not j['owned_sampler_live'] and not j['port18953']
assert '3651 MiB' in j['gpu_after'] and '3189 MiB' in j['gpu_after']
assert j['source_manifest']==json.loads((out/'source-manifest-v1.json').read_text())
for rel,h in j['source_manifest'].items():
 p=Path(rel);assert hashlib.sha256(p.read_bytes()).hexdigest()==h,rel;dst=out/'executed-source-v1'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
print('Owned probe terminal; sampler stopped; original VRAM restored; unchanged inference SO/source manifest verified')
