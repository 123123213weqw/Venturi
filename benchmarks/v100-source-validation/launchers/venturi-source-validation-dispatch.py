from pathlib import Path
import json,hashlib,shlex,subprocess,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
root=Path.cwd();out=root/'benchmarks/v100-source-validation';remote='/home/data/wangyue/repos/codex-build/Venturi/source-validation-src-v1'
previous=root/'benchmarks/v100-mtp-draft-residual'
pins=json.loads((previous/'prior-reference-pins.json').read_text())
for name in ['source-sha256.json','evidence-sha256.json','evidence-check.json']:
 p=previous/name;pins[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'prior-reference-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
audit=json.loads((previous/'prior-capsule-audit.json').read_text());audit['v100-mtp-draft-residual']={'frozen':True};assert len(audit)==39
(out/'prior-capsule-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
r=subprocess.run(ns['SSH'],input=f"import pathlib\np=pathlib.Path({remote!r});p.mkdir()\nprint(p)\n",text=True,capture_output=True,timeout=60);r.check_returncode()
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
source=r'''import pathlib,subprocess,os,json,time,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/source-validation-src-v1';out=root/'artifacts/venturi-source-validation-probe-v1';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,name,timeout=600):
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+chr(10))
 with (out/name).open('wb') as log:
  proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=proc.wait(timeout=timeout)
  except BaseException:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
def identity(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
sampler=None;samplerlog=None
try:
 EXPECTED_SOURCE
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 assert actual==expected,'Source changed before remote CPU compile'
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 previous=root/'artifacts/venturi-mtp-draft-residual-fixtures-v1';library=(previous/'library.txt').read_text().strip();assert sha(pathlib.Path(library))=='6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2'
 (out/'library.txt').write_text(library+chr(10))
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json'
 before={str(p):identity(p) for p in [target,draft]};(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-before.txt')
 run(['/usr/bin/strace','--version'],'strace-version.txt')
 run(['g++','--version'],'gpp-version.txt')
 run([py,'-m','pytest','-q','tests/test_source_validation_trace.py','--tb=short'],'parser-tests.log')
 run(['g++','-O2','-std=c++17',str(src/'tools/projection/source_validation.cpp'),'-o',str(out/'source-validation')],'cpu-build.log')
 (out/'cpu-binary-sha256.txt').write_text(sha(out/'source-validation')+chr(10))
 for name,path in [('target',target),('draft',draft)]:run([str(out/'source-validation'),str(path)],name+'-microbench.json')
 samplerlog=(out/'gpu-telemetry.csv').open('wb');sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv','-lms','100'],stdout=samplerlog,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'sampler.json').write_text(json.dumps(dict(pid=sampler.pid,started_unix=time.time()),indent=2)+chr(10))
 cmd=[py,'tools/probe_native_source_validation.py',str(target),'--mtp-artifact',str(draft),'--library',library,'--case',str(case)]
 run(cmd+['--repeats','3','--output',str(out/'untraced.json')],'untraced.log',timeout=1200)
 run(['/usr/bin/strace','-f','-qq','-ttt','-T','-yy','-e','trace=fstat,stat,newfstatat,statx','-o',str(out/'metadata.strace')]+cmd+['--repeats','1','--instrumented','--output',str(out/'traced.json')],'traced.log',timeout=1200)
 run([py,'tools/summarize_native_source_validation.py',str(out/'metadata.strace'),str(out/'traced.json'),'--output',str(out/'syscall-report.json')],'summarize.log')
 parsed=json.loads((out/'syscall-report.json').read_text());assert parsed['model_outputs_exact'] and not parsed['failures'] and not parsed['checkpoint_calls'].get('unclassified',0)
 assert parsed['checkpoint_calls']['descriptor']==parsed['checkpoint_calls']['pathname']>1000
 after={str(p):identity(p) for p in [target,draft]};assert after==before;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,engine_unchanged=True,no_source_guards_removed=True,library_sha256=sha(pathlib.Path(library)),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 if sampler is not None:
  os.killpg(sampler.pid,signal.SIGTERM);sampler.wait(timeout=20);samplerlog.close()
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''.replace(' EXPECTED_SOURCE',' expected='+repr(expected))
ns['dispatch']('venturi-source-validation-probe-v1',source)
