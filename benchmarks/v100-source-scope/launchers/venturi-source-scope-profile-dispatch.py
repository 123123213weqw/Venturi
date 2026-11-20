from pathlib import Path
import subprocess,shlex,json,hashlib,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
root=Path.cwd();out=root/'benchmarks/v100-source-scope';remote='/home/data/wangyue/repos/codex-build/Venturi/source-scope-src-v4'
r=subprocess.run(ns['SSH'],input=f"import pathlib\np=pathlib.Path({remote!r});p.mkdir()\n",capture_output=True,text=True,timeout=60);r.check_returncode()
cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
for pattern in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',pattern]
cmd+=['src','tests','tools','scripts','docs','README.md','pyproject.toml','AGENTS.md','WZU_Server:'+remote+'/']
for attempt in range(2):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (out/'sync-v4.log').open('a') as f:f.write(str(attempt)+'\n'+r.stdout+r.stderr)
 if not r.returncode:break
 if attempt==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};(out/'source-manifest-v4.json').write_text(json.dumps(expected,indent=2)+'\n')
old=Path('/tmp/venturi-source-validation-probe-v1-worker.py').read_text();source=old[:old.index('sampler=None;')].replace('source-validation-src-v1','source-scope-src-v4').replace('venturi-source-validation-probe-v1','venturi-source-scope-profile-v5')
source+='try:\n expected='+repr(expected)+r'''
 gate=root/'artifacts/venturi-source-scope-campaign-v4';deadline=time.monotonic()+7200
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];p=pathlib.Path(f'/proc/{pid}/cmdline');assert p.exists() and str(gate/'worker.py').encode() in p.read_bytes(),'Prerequisite missing without terminal marker'
  assert time.monotonic()<deadline,'Verified live prerequisite bounded wait elapsed';time.sleep(5)
 assert (gate/'exit-code.txt').read_text().strip()=='0','Campaign failed'
 campaign=json.loads((gate/'campaign-report.json').read_text());assert campaign['completed'] and campaign['model_qualified']
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 fixtures=root/'artifacts/venturi-source-scope-fixtures-v2';libs={0:pathlib.Path((fixtures/'default-library.txt').read_text().strip()),1:pathlib.Path((fixtures/'scoped-library.txt').read_text().strip())}
 for enabled,lib in libs.items():
  build=json.loads(lib.with_name('build.json').read_text());assert build['resident_source_scope']==bool(enabled) and sha(lib)==build['library_sha256']
  for rel,h in build['source_sha256'].items():assert sha(src/'src/venturi/ops/csrc'/rel)==h
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json'
 before={str(p):identity(p) for p in [target,draft]};(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 run([py,'-m','pytest','-q','tests/test_source_validation_trace.py','--tb=short'],'parser-tests.log')
 run(['/usr/bin/strace','-f','-qq','-ttt','-T','-yy','-e','trace=fstat,stat,newfstatat,statx','-o',str(out/'scoped-metadata.strace'),py,'tools/probe_native_source_validation.py',str(target),'--mtp-artifact',str(draft),'--library',str(libs[1]),'--case',str(case),'--repeats','1','--instrumented','--output',str(out/'scoped-metadata.json')],'scoped-metadata.log',timeout=900)
 run([py,'tools/summarize_native_source_validation.py',str(out/'scoped-metadata.strace'),str(out/'scoped-metadata.json'),'--output',str(out/'scoped-syscalls.json')],'scoped-syscalls.log')
 raw=json.loads((out/'scoped-metadata.json').read_text());counts=json.loads((out/'scoped-syscalls.json').read_text());assert not raw['source_guards_unchanged'] and raw['source_integrity_predicate_preserved']
 assert counts['engine_change'] and counts['source_check_cadence_changed'] and not counts['failures'] and counts['model_outputs_exact']
 assert counts['checkpoint_calls']['descriptor']==counts['checkpoint_calls']['pathname']<6674
 for scope,library in libs.items():
  for mode in ['mtp2-c1','mtp-batch-c8']:
   name='scope'+str(scope)+'-'+mode
   if mode=='mtp2-c1':cmd=[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--case',str(case),'--depth','2','--count','1','--recovery','prefix']
   else:cmd=[py,'tools/profile_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--case',str(case),'--count','8']
   cmd+=['--projection-profile','q8-residual-column-mlp','--mtp-draft-projection-profile','q8-residual','--output',str(out/(name+'-metadata.json'))]
   run(['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--cuda-event-trace=false','--cuda-graph-trace=node','-o',str(out/name)]+cmd,name+'-profile.log',timeout=900)
   run(['/usr/local/bin/nsys','export','--type=sqlite','--output',str(out/(name+'.sqlite')),str(out/(name+'.nsys-rep'))],name+'-export.log')
   run([py,'tools/summarize_native_parallel_trace.py',str(out/(name+'.sqlite')),'--output',str(out/(name+'-attribution.json'))],name+'-attribution.log')
   metadata=json.loads((out/(name+'-metadata.json')).read_text());assert metadata['source_validation_policy'].startswith('resident-forward')==(scope==1)
  run([py,'tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',sha(library),'--output',str(out/('scope'+str(scope)+'-resources.json'))],'scope'+str(scope)+'-resources.log')
 after={str(p):identity(p) for p in [target,draft]};assert before==after;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 retained={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']};(out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library_sha256_by_scope={str(k):sha(v) for k,v in libs.items()},source_check_counts_reduced=True,profiled_separately_from_unprofiled_campaign=True,ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-source-scope-profile-v5',source)
