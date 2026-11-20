from pathlib import Path
import json,hashlib,subprocess,shlex,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-native-head-gdn';remote='/home/data/wangyue/repos/codex-build/Venturi/native-head-gdn-src-v1'
r=subprocess.run(ns['SSH'],input=f"import pathlib,subprocess\np=pathlib.Path({remote!r});p.mkdir()\nprint(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout)\n",text=True,capture_output=True,timeout=60);r.check_returncode();(out/'pre-sync-observation.log').write_text(r.stdout+r.stderr);print(r.stdout)
cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
for p in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',p]
cmd+=['src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/']
for i in range(2):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (out/'sync-v1.log').open('a') as f:f.write(str(i)+'\n'+r.stdout+r.stderr)
 if not r.returncode:break
 if i==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};(out/'source-manifest-v1.json').write_text(json.dumps(expected,indent=2)+'\n')
for rel,h in expected.items():
 p=root/rel;dst=out/'executed-source-v1'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
old=(root/'benchmarks/v100-mtp-draft-residual/v100/fixtures-v1/worker.py').read_text()
source=old[:old.index('try:\n expected=')].replace('mtp-draft-residual-src-v1','native-head-gdn-src-v1').replace('venturi-mtp-draft-residual-fixtures-v1','venturi-native-head-gdn-fixtures-v1')
source+='try:\n expected='+repr(expected)+r'''
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'pre-build-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 buildcmd="from venturi.ops.build import build_blocks_backend;from pathlib import Path;p=build_blocks_backend("+repr(str(out/'build'))+",nvcc='/usr/local/cuda/bin/nvcc',architecture='sm70');Path("+repr(str(out/'library.txt'))+").write_text(str(p))"
 run([py,'-c',buildcmd],'build.log',timeout=900)
 library=(out/'library.txt').read_text().strip();env['VENTURI_NATIVE_HEAD_GDN_LIBRARY']=library
 run([py,'-m','pytest','-x','-q','tests/test_native_head_gdn.py','--tb=short'],'focused-tests.log',timeout=900)
 CONTROL_ENV
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=1200)
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
control=old[old.index(" baseline=root/'artifacts/venturi-projection-fusion-model-v3"):old.index(" run([py,'-m','pytest','-q','--tb=short'],'full-tests.log'")]
source=source.replace(' CONTROL_ENV',control.rstrip())
(out/'fixtures-v1-launcher.py').write_text(Path(__file__).read_text())
ns['dispatch']('venturi-native-head-gdn-fixtures-v1',source)
