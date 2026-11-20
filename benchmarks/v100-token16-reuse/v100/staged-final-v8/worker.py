import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/token16-src-v2';out=root/'artifacts/venturi-token16-staged-final-v8';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run(cmd,name,timeout=300):
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/name).open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:stop(p);raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
try:
 gate=root/'artifacts/venturi-token16-staged-campaign-v7';deadline=time.monotonic()+3600
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];cmdline=pathlib.Path(f'/proc/{pid}/cmdline');assert cmdline.exists() and str(gate/'worker.py').encode() in cmdline.read_bytes(),'Pinned predecessor absent';assert time.monotonic()<deadline,'Bounded predecessor wait elapsed';time.sleep(10)
 assert (gate/'exit-code.txt').read_text().strip()=='0';campaign=json.loads((gate/'campaign-report.json').read_text());assert campaign['completed']
 previous=root/'artifacts/venturi-token16-staged-study-v5';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text());baseline=str(root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so')
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'reused-fixture-tests.json').write_text(json.dumps(dict(path=str(previous),sha256=hashlib.sha256((previous/'full-tests.log').read_bytes()).hexdigest(),scope='Exact native library and nine dependency hashes; full fixture suite reused. Whole-call profiles remain separate from unprofiled matrices.'),indent=2)+'\n')
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';nsys='/usr/local/bin/nsys';case=json.loads((root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json').read_text());(out/'profile-case.json').write_text(json.dumps(case,indent=2)+'\n');traces=[]
 if campaign['operator_promoted']:
  for name,cmd in [
   ('mtp2-prefix-c1',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','1']),
   ('ordinary256-c8',[py,'tools/profile_qwen_batch.py',str(target),'--library',library,'--case',str(out/'profile-case.json'),'--mode','batch','--projection-profile','q8-residual','--projection-rows','256','--count','8']),
   ('mtp-batch-c8',[py,'tools/profile_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--count','8'])]:
   prefix=out/name;cmd+=['--output',str(out/(name+'-metadata.json'))]
   run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),*cmd],name+'-profile.log',timeout=300)
   run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=120)
   run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=90)
   traces+=['--trace',name+'='+str(prefix)+'.sqlite']
 control=root/'artifacts/venturi-batched-mtp-model-v2/build/e94e01699eaad5b0/libventuri_blocks.so'
 assert hashlib.sha256(control.read_bytes()).hexdigest()=='f07bf75a975b963911aff5fefd5614dbf0b8b5b74e499f5525b6ad5d5317369a'
 import re,collections
 sass={}
 for label,so in [('candidate',library)]:
  prefix=out/label
  run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_q8_token16.py',str(target),'--library',so,'--output',str(out/(label+'-metadata.json'))],label+'-profile.log',timeout=300)
  run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],label+'-export.log',timeout=120)
  run([py,'tools/summarize_native_runtime_trace.py',str(prefix)+'.sqlite','--output',str(out/(label+'-attribution.json'))],label+'-attribution.log',timeout=90)
  run([py,'tools/inspect_cuda_occupancy.py','--library',so,'--expected-sha256',hashlib.sha256(pathlib.Path(so).read_bytes()).hexdigest(),'--trace',label+'='+str(prefix)+'.sqlite','--output',str(out/(label+'-resources.json'))],label+'-resources.log',timeout=180)
  run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',so],label+'-sass.txt',timeout=120)
  raw=(out/(label+'-sass.txt')).read_text();items={};snippets=[]
  for match in re.finditer(r'Function : (\S+)(.*?)(?=\n\s*Function :|\Z)',raw,re.S):
   symbol,body=match.groups()
   if not any(v in symbol for v in ['3vdt3dotILi8ELi2ELi16E','3vqf9tiled_dotILi8ELi2ELi16E','3vdt12staged16_dotILi2ELi8E','3vqf12staged16_dotILi2ELi8E']):continue
   ops=collections.Counter(re.findall(r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?P[0-9]+\s+)?([A-Z][A-Z0-9_.]*)',body,re.M));items[symbol]=dict(static_instruction_count=sum(ops.values()),opcodes=dict(ops));snippets.append('Function : '+symbol+body)
  assert len(items)==(4 if label=='candidate' else 2),(label,list(items))
  (out/(label+'-target-sass.txt')).write_text('\n'.join(snippets));sass[label]=dict(library_sha256=hashlib.sha256(pathlib.Path(so).read_bytes()).hexdigest(),functions=items)
 old=root/'artifacts/venturi-token16-final-v4';refs={}
 for name in ['control-metadata.json','control-attribution.json','control-resources.json','candidate-metadata.json','candidate-attribution.json','candidate-resources.json','target-sass-instructions.json','remote-retained-files.json','executed-source.json']:
  p=old/name;refs[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (out/'reused-control-and-natural-operator-profiles.json').write_text(json.dumps(refs,indent=2)+'\n')
 (out/'target-sass-instructions.json').write_text(json.dumps(dict(scope='Static disassembly, not executed instructions or hardware counters.',libraries=sass),indent=2)+'\n')
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),*traces,'--output',str(out/'resources.json')],'resources.log',timeout=180)
 reused={}
 for folder,names in [
  ('venturi-native-mtp2-full-profile-v14',['mtp2-prefix-full-c1-metadata.json','mtp2-prefix-full-c1-attribution.json','resources.json','remote-retained-files.json','executed-source.json']),
  ('venturi-batched-mtp-final-v4',['ordinary256-control-c8-metadata.json','ordinary256-control-c8-attribution.json','control-resources.json','remote-retained-files.json','executed-source.json']),
  ('venturi-batched-mtp-sdk-final-v9',['mtp-batch-c8-metadata.json','mtp-batch-c8-attribution.json','remote-retained-files.json','executed-source.json'])]:
  old=root/'artifacts'/folder
  for name in names:
   p=old/name;reused[folder+'/'+name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (out/'reused-control-profiles.json').write_text(json.dumps(reused,indent=2)+'\n')
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep'] or p.name in ['control-sass.txt','candidate-sass.txt']},indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
