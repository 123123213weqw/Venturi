from pathlib import Path
import json,hashlib
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
root=Path.cwd();cap=root/'benchmarks/v100-gdn-head-pilot'
expected=json.loads((cap/'source-manifest-v4.json').read_text())
base=Path('/tmp/venturi-gdn-head-pilot-fixtures-v7-worker.py').read_text().split('try:\n expected=')[0].replace("out=root/'artifacts/venturi-gdn-head-pilot-fixtures-v7'", "out=root/'artifacts/venturi-gdn-head-pilot-trained-v8'")
identity=json.loads((root/'benchmarks/v100-source-scope/reused-full-payload-pins.json').read_text())
source=base+'try:\n expected='+repr(expected)+'\n pinned='+repr(identity)+r'''
 fixture=root/'artifacts/venturi-gdn-head-pilot-fixtures-v7';deadline=time.monotonic()+180
 while not (fixture/'exit-code.txt').exists():
  pid=json.loads((fixture/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(fixture/'worker.py').encode() in q.read_bytes();assert time.monotonic()<deadline;time.sleep(2)
 assert (fixture/'exit-code.txt').read_text().strip()=='0'
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 build=json.loads((fixture/'build.json').read_text());blocks=pathlib.Path(build['blocks_library']);library=pathlib.Path(build['library']);assert sha(blocks)==build['blocks_library_sha256']=='6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2';assert sha(library)==build['library_sha256']=='202a38580846334ac9d08893b4f85064f6f5eae8f65f451c2200cfe0c78fa8e1'
 assert sha(src/'tools/projection/gdn_head_pilot.cpp')==build['host_cpp_sha256']
 (out/'build.json').write_text(json.dumps(build,indent=2)+chr(10))
 def identity(p):
  s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';bf16=root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'
 before={str(target):identity(target)};assert before[str(target)]==pinned['identity'][str(target)]
 (out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 source_manifest=json.loads((bf16/'venturi-source.json').read_text());assert source_manifest['status']=='complete' and source_manifest['revision']=='fc05daec18b0a78c049392ed2e771dde82bdf654'
 index=json.loads((bf16/'model.safetensors.index.json').read_text())['weight_map'];selected={v for k,v in index.items() if any(k.startswith(f'model.language_model.layers.{layer}.linear_attn.') or k==f'model.language_model.layers.{layer}.input_layernorm.weight' for layer in (0,62))}
 selected.update(['venturi-source.json','config.json','model.safetensors.index.json']);source_before={str(bf16/name):identity(bf16/name) for name in selected};digests={}
 for name in selected:
  digest=hashlib.sha256()
  with (bf16/name).open('rb') as f:
   while chunk:=f.read(8*1024*1024):digest.update(chunk)
  digests[name]=digest.hexdigest()
  if name.endswith('.safetensors'):
   original=next(f for f in source_manifest['files'] if f['name']==name);assert digests[name]==original['actual_sha256']==original['lfs_sha256'],name
 (out/'verified-source-files.json').write_text(json.dumps(dict(official_revision=source_manifest['revision'],fresh_sha256=digests,identity_before=source_before,source_manifest=source_manifest),indent=2)+chr(10))
 (out/'reused-payload-pins.json').write_text(json.dumps(pinned,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-before.txt')
 run([py,'tools/benchmark_gdn_head_tp.py',str(target),'--source',str(bf16),'--blocks',str(blocks),'--host-library',str(library),'--output-directory',str(out/'measurement')],'benchmark.log',timeout=1800)
 report=json.loads((out/'measurement/report.json').read_text());assert report['completed'] and len(report['cells'])==6
 source_after={p:identity(pathlib.Path(p)) for p in source_before};assert source_after==source_before;(out/'source-after.json').write_text(json.dumps(source_after,indent=2)+chr(10))
 after={str(target):identity(target)};assert after==before
 (out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,model_integration=False,model_tps_claim=False,integration_gate=report['integration_gate'],ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
(cap/'trained-v8-launcher.py').write_text(Path(__file__).read_text())
for rel,h in expected.items():
 p=root/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,rel
 dst=cap/'executed-source-v4'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
n['dispatch']('venturi-gdn-head-pilot-trained-v8',source)
