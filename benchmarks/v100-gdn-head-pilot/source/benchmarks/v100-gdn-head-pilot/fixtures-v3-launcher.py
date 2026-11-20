from pathlib import Path
import subprocess,shlex,json,hashlib,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
root=Path.cwd();out=root/'benchmarks/v100-gdn-head-pilot';remote='/home/data/wangyue/repos/codex-build/Venturi/gdn-head-pilot-src-v3'
prior=root/'benchmarks/v100-source-scope';pins=json.loads((prior/'prior-reference-pins.json').read_text())
for name in ['source-sha256.json','evidence-sha256.json','evidence-check.json']:
 p=prior/name;pins[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'prior-reference-pins.json').write_text(json.dumps(pins,indent=2)+'\n');audit=json.loads((prior/'prior-capsule-audit.json').read_text());audit['v100-source-scope']={'frozen':True};assert len(audit)==41
(out/'prior-capsule-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
r=subprocess.run(ns['SSH'],input=f"import pathlib,subprocess\np=pathlib.Path({remote!r});p.mkdir()\nprint(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout)\n",capture_output=True,text=True,timeout=60);r.check_returncode();(out/'pre-sync-observation.log').write_text(r.stdout+r.stderr);print(r.stdout)
cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
for pattern in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',pattern]
cmd+=['src','tests','tools','scripts','docs','README.md','pyproject.toml','AGENTS.md','WZU_Server:'+remote+'/']
for attempt in range(2):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (out/'sync-v3.log').open('a') as f:f.write(str(attempt)+'\n'+r.stdout+r.stderr)
 if not r.returncode:break
 if attempt==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};(out/'source-manifest-v3.json').write_text(json.dumps(expected,indent=2)+'\n')
old=Path('/tmp/venturi-source-validation-probe-v1-worker.py').read_text();source=old[:old.index('sampler=None;')].replace('source-validation-src-v1','gdn-head-pilot-src-v3').replace('venturi-source-validation-probe-v1','venturi-gdn-head-pilot-fixtures-v3')
source+='try:\n expected='+repr(expected)+r'''
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'pre-build-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 blocks=root/'artifacts/venturi-mtp-draft-residual-fixtures-v1/build/089e914d53d145f0/libventuri_blocks.so';assert sha(blocks)=='6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2'
 build=json.loads(blocks.with_name('build.json').read_text())
 for rel,h in build['source_sha256'].items():assert sha(src/'src/venturi/ops/csrc'/rel)==h
 library=out/'libventuri_gdn_head_host.so'
 run(['g++','-O3','-std=c++20','-shared','-fPIC','-I/usr/local/cuda/include',str(src/'tools/projection/gdn_head_pilot.cpp'),'-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-ldl','-o',str(library)],'host-build.log')
 run(['g++','--version'],'compiler-version.txt')
 (out/'library.txt').write_text(str(library)+chr(10));(out/'build.json').write_text(json.dumps(dict(library=str(library),library_sha256=sha(library),host_cpp_sha256=sha(src/'tools/projection/gdn_head_pilot.cpp'),blocks_library=str(blocks),blocks_library_sha256=sha(blocks),no_new_cuda_kernels=True),indent=2)+chr(10))
 env.update(VENTURI_GDN_HEAD_HOST=str(library),VENTURI_GDN_HEAD_BLOCKS=str(blocks))
 run([py,'-m','pytest','-x','-q','tests/test_gdn_head_pilot.py','--tb=short'],'focused-tests.log',timeout=900)
 run([py,'-m','pytest','-q','--tb=short'],'cpu-and-pilot-regression.log',timeout=900)
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library_sha256=sha(library),blocks_sha256=sha(blocks),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-gdn-head-pilot-fixtures-v3',source)
