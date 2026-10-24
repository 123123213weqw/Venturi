import pathlib,subprocess,os,json,time,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/mtp-draft-residual-src-v1';out=root/'artifacts/venturi-mtp-draft-residual-model-v2';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,name,timeout=1200):
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+chr(10))
 with (out/name).open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
try:
 gate=root/'artifacts/venturi-mtp-draft-residual-fixtures-v1';deadline=time.monotonic()+2400
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];cmd=pathlib.Path(f'/proc/{pid}/cmdline');assert cmd.exists() and str(gate/'worker.py').encode() in cmd.read_bytes(),'Prerequisite worker actually missing'
  assert time.monotonic()<deadline,'Prerequisite still live after bounded wait';time.sleep(3)
 assert (gate/'exit-code.txt').read_text().strip()=='0','Prerequisite failed; preserve failure'
 expected=json.loads((gate/'executed-source.json').read_text())
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 library=(gate/'library.txt').read_text().strip();(out/'library.txt').write_text(library+chr(10))
 (out/'reused-build-and-fixtures.json').write_text(json.dumps(dict(path=str(gate),full_tests_sha256=sha(gate/'full-tests.log'),focused_tests_sha256=sha(gate/'focused-tests.log'),library_sha256=sha(pathlib.Path(library))),indent=2)+chr(10))
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';prior=root/'artifacts/venturi-native-column-mlp-model-v2/qualification/report.json'
 assert sha(prior)=='904bace8b87a2047191e4fdeab66a420dfacb100d5d9686beb1ef52ac0d655f7'
 cmd=[py,'tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(draft),'--library',library,'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual-column-mlp','--draft-tokens','2','--recovery','prefix','--prior-target-report',str(prior),'--prior-target-report-sha256',sha(prior)]
 for profile in ['fp16','q8-residual']:
  run(cmd+['--mtp-draft-projection-profile',profile,'--output-directory',str(out/(profile+'-qualification'))],profile+'-qualification.log')
  report=json.loads((out/(profile+'-qualification/report.json')).read_text());assert report['completed'] and len(report['block_reference'])==2 and len(report['recurrence_reference'])==3 and len(report['cases'])==4
  assert all(c['passed'] for c in report['block_reference']+report['recurrence_reference'])
 run([py,'tools/qualify_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(draft),'--library',library,'--cases',str(cases),'--prior-target-report',str(prior),'--prior-target-report-sha256',sha(prior),'--projection-profile','q8-residual-column-mlp','--mtp-draft-projection-profile','q8-residual','--output-directory',str(out/'batch-qualification')],'batch-qualification.log')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library_sha256=sha(pathlib.Path(library)),fresh_independent_draft_profiles=['fp16','q8-residual'],unchanged_target_teacher_reused=True,ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
