from pathlib import Path
import subprocess,json,hashlib
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-column-tp';snap='column-tp-src-v3';remote='/home/data/wangyue/repos/codex-build/Venturi/'+snap
subprocess.run(ns['SSH'][:-1]+['mkdir -p '+remote],check=True,timeout=60)
ssh=' '.join(ns['SSH'][:-2]);flags=['--exclude='+v for v in ['.git/','target/','node_modules/','.env','.env.*','.venv*','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']]
r=subprocess.run(['rsync','-az',*flags,'-e',ssh,'src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/'],capture_output=True,text=True,timeout=180);(out/'sync-v3.log').write_text(r.stdout+r.stderr);r.check_returncode()
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};(out/'source-manifest-v3.json').write_text(json.dumps(expected,indent=2)+'\n')
base=Path('/tmp/venturi-column-tp-fixtures-v1-worker.py').read_text();head=base[:base.index('try:\n expected=')];head=head.replace("column-tp-src-v1","column-tp-src-v3").replace("artifacts/venturi-column-tp-fixtures-v1","artifacts/venturi-column-tp-mlp-v4")
body=r'''try:
 expected=EXPECTED
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert actual==expected,'Full source manifest mismatch before use'
 (out/'pre-execution-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 previous=root/'artifacts/venturi-column-tp-fixtures-v1';deadline=time.monotonic()+1200
 while not (previous/'exit-code.txt').exists():
  assert time.monotonic()<deadline,'Bounded predecessor wait elapsed';time.sleep(10)
 assert (previous/'exit-code.txt').read_text().strip()=='0'
 library=(previous/'library.txt').read_text().strip();build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 assert len(build['source_sha256'])==10
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'library.txt').write_text(library+chr(10))
 (out/'reused-build.json').write_text(json.dumps(dict(path=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),source_sha256=build['source_sha256'],reason='Only new benchmark/trace tooling and CPU tests; exact ten native dependencies, full fixture suite reused.'),indent=2)+chr(10))
 run([py,'-m','pytest','-x','-q','tests/test_q8_column_tp_tool.py','tests/test_ada_build.py','--tb=short'],'tool-tests.log',timeout=120)
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'
 cmd=[py,'tools/benchmark_q8_column_tp.py',str(target),'--library',library]
 run(cmd+['--output-directory',str(out/'unprofiled')],'benchmark.log',timeout=600)
 report=json.loads((out/'unprofiled/report.json').read_text());assert report['completed'] and len(report['cases'])==3
 nsys='/usr/local/bin/nsys';prefix=out/'mlp-profile'
 run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--cuda-graph-trace=node','--force-overwrite=false','--output',str(prefix),*cmd,'--output-directory',str(out/'profiled'),'--profile'],'profile.log',timeout=600)
 run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],'export.log',timeout=240)
 run([py,'tools/summarize_column_tp_trace.py',str(prefix)+'.sqlite','--output',str(out/'attribution.json')],'attribution.log',timeout=180)
 attr=json.loads((out/'attribution.json').read_text());assert len(attr['operators'])==6
 for cell in attr['operators']:
  _,width,mode=cell['name'].split('/');tokens=int(width[1:]);expected_bytes=tokens*(3*5120+4*17408 if mode=='row' else 6*5120)*12
  assert cell['transfers']['host_to_device']['bytes']==expected_bytes and cell['transfers']['device_to_host']['bytes']==expected_bytes,(cell['name'],cell['transfers'],expected_bytes)
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),'--output',str(out/'resources.json')],'resources.log',timeout=180)
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']},indent=2)+chr(10))
 (out/'counter-policy.txt').write_text(next(v for v in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in v)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-column-tp-mlp-v4',head+body.replace(' expected=EXPECTED',' expected='+repr(expected)))
