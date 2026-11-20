from pathlib import Path
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
source=r'''import pathlib,subprocess,os,json,time,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/mtp-draft-residual-src-v1';out=root/'artifacts/venturi-mtp-draft-residual-profile-v4';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,name,timeout=600):
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
 gate=root/'artifacts/venturi-mtp-draft-residual-campaign-v3';deadline=time.monotonic()+5400
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];cmd=pathlib.Path(f'/proc/{pid}/cmdline');assert cmd.exists() and str(gate/'worker.py').encode() in cmd.read_bytes(),'Prerequisite actually missing'
  assert time.monotonic()<deadline,'Bounded live prerequisite wait elapsed';time.sleep(3)
 assert (gate/'exit-code.txt').read_text().strip()=='0';campaign=json.loads((gate/'campaign-report.json').read_text());assert campaign['completed'] and campaign['model_qualified']
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==campaign['source_sha256']
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 previous=root/'artifacts/venturi-mtp-draft-residual-model-v2';library=(previous/'library.txt').read_text().strip();assert sha(pathlib.Path(library))==campaign['candidate_library_sha256'];(out/'library.txt').write_text(library+chr(10))
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json'
 commands=[]
 for profile,name in [('fp16','fp16-draft-mtp2-c1'),('q8-residual','residual-draft-mtp2-c1')]:
  commands.append((name,[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(draft),'--library',library,'--case',str(case),'--depth','2','--count','1','--recovery','prefix','--projection-profile','q8-residual-column-mlp','--mtp-draft-projection-profile',profile,'--output',str(out/(name+'-metadata.json'))]))
 commands.append(('residual-draft-mtp-batch-c8',[py,'tools/profile_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(draft),'--library',library,'--case',str(case),'--count','8','--projection-profile','q8-residual-column-mlp','--mtp-draft-projection-profile','q8-residual','--output',str(out/'residual-draft-mtp-batch-c8-metadata.json')]))
 for name,cmd in commands:
  run(['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--cuda-event-trace=false','--cuda-graph-trace=node','-o',str(out/name)]+cmd,name+'-profile.log')
  run(['/usr/local/bin/nsys','export','--type=sqlite','--output',str(out/(name+'.sqlite')),str(out/(name+'.nsys-rep'))],name+'-export.log')
  run([py,'tools/summarize_native_parallel_trace.py',str(out/(name+'.sqlite')),'--output',str(out/(name+'-attribution.json'))],name+'-attribution.log')
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',sha(pathlib.Path(library)),'--output',str(out/'resources.json')],'resources.log')
 c=json.loads(case.read_text())
 cmd=[py,'-m','venturi','qwen-native','generate',str(target),'--blocks-library',library,'--weight-mode','resident','--dtype','fp16','--device','cuda:0','--tensor-parallel-device','cuda:1','--tensor-parallel-transport','host','--projection-profile','q8-residual-column-mlp','--max-tokens','32','--capacity','256','--token-ids',','.join(map(str,c['prompt_token_ids'])),'--max-new-tokens','32','--eos-token-ids','','--mtp-artifact',str(draft),'--mtp-draft-tokens','2','--mtp-recovery','prefix','--mtp-draft-projection-profile','q8-residual']
 run(cmd,'actual-cli.json')
 cli=json.loads((out/'actual-cli.json').read_text());assert cli['generated_ids']==c['expected_generated_ids'] and cli['mtp_draft_projection_profile']=='q8-residual';assert cli['native_memory']['mtp_draft_activation_quantization_passes']==2 and cli['native_memory']['mtp_draft_primary_projection_workspace_bytes']>0
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
 retained={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.suffix in ('.sqlite','.nsys-rep')}
 (out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,ended_unix=time.time(),actual_cli_fixed_tokens_and_draft_profile_passed=True,library_sha256=sha(pathlib.Path(library))),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-mtp-draft-residual-profile-v4',source)
