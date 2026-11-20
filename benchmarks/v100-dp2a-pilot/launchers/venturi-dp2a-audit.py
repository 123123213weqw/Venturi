from pathlib import Path
import json,subprocess,hashlib,shutil
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
root=Path.cwd();out=root/'benchmarks/v100-dp2a-pilot'
code=r'''import pathlib,subprocess,json,hashlib,time
root=pathlib.Path('/home/data/wangyue');workers={}
for label in ['fixtures-v1','campaign-v2']:
 p=root/('artifacts/venturi-dp2a-pilot-'+label);workers[label]=dict(process=json.loads((p/'process.json').read_text()),exit_code=int((p/'exit-code.txt').read_text()))
processes=subprocess.run(['ps','-eo','pid,pgid,args'],capture_output=True,text=True,check=True).stdout
owned=[l for l in processes.splitlines() if 'venturi-dp2a-pilot' in l and '<defunct>' not in l]
samplers=[l for l in processes.splitlines() if 'nvidia-smi' in l and '-lms' in l]
source={}
for version in [1,2]:
 p=root/f'repos/codex-build/Venturi/dp2a-pilot-src-v{version}'
 source[str(p)]={str(q.relative_to(p)):hashlib.sha256(q.read_bytes()).hexdigest() for d in ['src','tests','tools'] for q in (p/d).rglob('*') if q.is_file() and '__pycache__' not in q.parts and q.suffix!='.pyc'}
print(json.dumps(dict(ended_unix=time.time(),workers=workers,owned_processes=owned,remaining_periodic_samplers=samplers,gpu_after=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout,counter_policy=pathlib.Path('/proc/driver/nvidia/params').read_text(),snapshots=source),indent=2))
'''
p=subprocess.run(ns['SSH'],input=code,text=True,capture_output=True,timeout=90);p.check_returncode();result=json.loads(p.stdout)
assert all(v['exit_code']==0 for v in result['workers'].values()) and not result['owned_processes'] and not result['remaining_periodic_samplers']
assert '3651 MiB' in result['gpu_after'] and '3189 MiB' in result['gpu_after']
(out/'terminal-audit.json').write_text(json.dumps(result,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for version in [1,2]:
 m=json.loads((out/f'source-manifest-v{version}.json').read_text());remote=f'/home/data/wangyue/repos/codex-build/Venturi/dp2a-pilot-src-v{version}';assert m==result['snapshots'][remote]
 for rel,h in m.items():
  dst=out/f'executed-source-v{version}'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  candidates=[root/rel,root/'benchmarks/v100-native-column-mlp/source'/rel]
  src=next((p for p in candidates if p.is_file() and sha(p)==h),None);assert src,rel;shutil.copy2(src,dst)
print('Two terminal workers; original VRAM restored; both executed source snapshots preserved')
