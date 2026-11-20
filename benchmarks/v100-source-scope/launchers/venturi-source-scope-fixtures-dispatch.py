from pathlib import Path
import subprocess,shlex,json,hashlib,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
root=Path.cwd();out=root/'benchmarks/v100-source-scope';remote='/home/data/wangyue/repos/codex-build/Venturi/source-scope-src-v1'
prior=root/'benchmarks/v100-source-validation';pins=json.loads((prior/'prior-reference-pins.json').read_text())
for name in ['source-sha256.json','evidence-sha256.json','evidence-check.json']:
 p=prior/name;pins[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'prior-reference-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
audit=json.loads((prior/'prior-capsule-audit.json').read_text());audit['v100-source-validation']={'frozen':True};assert len(audit)==40
(out/'prior-capsule-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
code=f"import pathlib,subprocess,json\np=pathlib.Path({remote!r});p.mkdir()\nprint(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout)"
r=subprocess.run(ns['SSH'],input=code,text=True,capture_output=True,timeout=60);(out/'pre-sync-observation.log').write_text(r.stdout+r.stderr);r.check_returncode();print(r.stdout)
cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
for pattern in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',pattern]
cmd+=['src','tests','tools','scripts','docs','README.md','pyproject.toml','AGENTS.md','WZU_Server:'+remote+'/']
for attempt in range(2):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (out/'sync-v1.log').open('a') as f:f.write(str(attempt)+'\n'+r.stdout+r.stderr)
 if not r.returncode:break
 if attempt==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'source-manifest-v1.json').write_text(json.dumps(expected,indent=2)+'\n')
old=Path('/tmp/venturi-mtp-draft-residual-fixtures-v1-worker.py').read_text()
source=old[:old.index('try:\n expected=')].replace('mtp-draft-residual-src-v1','source-scope-src-v1').replace('venturi-mtp-draft-residual-fixtures-v1','venturi-source-scope-fixtures-v1')
source+='try:\n expected='+repr(expected)+'''
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 assert actual==expected,'Full source mismatch before compilation'
 (out/'pre-build-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 for scope,name in [(False,'default'),(True,'scoped')]:
  buildcmd="from venturi.ops.build import build_blocks_backend;from pathlib import Path;p=build_blocks_backend("+repr(str(out/'build'))+",nvcc='/usr/local/cuda/bin/nvcc',architecture='sm70',resident_source_scope="+str(scope)+");Path("+repr(str(out/(name+'-library.txt')))+").write_text(str(p)+chr(10))"
  run([py,'-c',buildcmd],name+'-build.log',timeout=900)
 library=(out/'scoped-library.txt').read_text().strip();default=(out/'default-library.txt').read_text().strip();(out/'library.txt').write_text(library+chr(10))
 control=str(root/'artifacts/venturi-mtp-draft-residual-fixtures-v1/build/089e914d53d145f0/libventuri_blocks.so')
 env.update(VENTURI_SOURCE_SCOPE_LIBRARY=library,VENTURI_SOURCE_SCOPE_CONTROL_LIBRARY=control,VENTURI_SOURCE_SCOPE_DEFAULT_LIBRARY=default)
 run([py,'-m','pytest','-x','-q','tests/test_resident_source_scope.py','--tb=short'],'focused-tests.log',timeout=600)
'''
# Reuse the complete previously qualified test-environment selection, not its old worker source.
full=old[old.index(" baseline=root/"):old.index(" (out/'executed-source.json')")]
full=full.replace("env.update(VENTURI_MTP_DRAFT_PROFILE_LIBRARY", "env.update(VENTURI_MTP_DRAFT_PROFILE_LIBRARY")
source+=" env.update(VENTURI_MTP_DRAFT_PROFILE_LIBRARY=library,VENTURI_MTP_DRAFT_CONTROL_LIBRARY=str(root/'artifacts/venturi-native-column-mlp-fixtures-v1/build/77588f610f53223e/libventuri_blocks.so'))\n"+full
source+='''
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),default_library=default,default_library_sha256=hashlib.sha256(pathlib.Path(default).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-source-scope-fixtures-v1',source)
