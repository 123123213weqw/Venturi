import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/activation-words-src-v1';out=root/'artifacts/venturi-activation-words-micro-profile-v2';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 gate=root/'artifacts/venturi-activation-words-study-v1';deadline=time.monotonic()+1500
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];c=pathlib.Path(f'/proc/{pid}/cmdline');assert c.exists() and str(gate/'worker.py').encode() in c.read_bytes(),'Pinned study handle absent';assert time.monotonic()<deadline,'Observation deadline reached; revalidate same handle, never relaunch';time.sleep(10)
 assert (gate/'exit-code.txt').read_text().strip()=='0'
 library=(gate/'library.txt').read_text().strip();control=root/'artifacts/venturi-batched-mtp-model-v2/build/e94e01699eaad5b0/libventuri_blocks.so';assert hashlib.sha256(control.read_bytes()).hexdigest()=='f07bf75a975b963911aff5fefd5614dbf0b8b5b74e499f5525b6ad5d5317369a'
 (out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 traces=[];sass={};nsys='/usr/local/bin/nsys'
 for label,so in [('control',str(control)),('candidate',library)]:
  prefix=out/label
  run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_q8_activation_words.py',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'),'--library',so,'--output',str(out/(label+'-metadata.json'))],label+'-profile.log',timeout=300)
  run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],label+'-export.log',timeout=120)
  run([py,'tools/summarize_native_runtime_trace.py',str(prefix)+'.sqlite','--output',str(out/(label+'-attribution.json'))],label+'-attribution.log',timeout=90)
  run([py,'tools/inspect_cuda_occupancy.py','--library',so,'--expected-sha256',hashlib.sha256(pathlib.Path(so).read_bytes()).hexdigest(),'--trace',label+'='+str(prefix)+'.sqlite','--output',str(out/(label+'-resources.json'))],label+'-resources.log',timeout=180)
  run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',so],label+'-sass.txt',timeout=120)
  import re,collections
  raw=(out/(label+'-sass.txt')).read_text();items={}
  for match in re.finditer(r'Function : (\S+)(.*?)(?=\n\s*Function :|\Z)',raw,re.S):
   symbol,body=match.groups()
   if not any(v in symbol for v in ['3vdt3dotILi4ELi2ELi8E','3vqf9tiled_dotILi4ELi2ELi8E']):continue
   ops=collections.Counter(re.findall(r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?P[0-9]+\s+)?([A-Z][A-Z0-9_.]*)',body,re.M));items[symbol]=dict(static_instruction_count=sum(ops.values()),opcodes=dict(ops))
  assert len(items)==2,(label,list(items));sass[label]=dict(library_sha256=hashlib.sha256(pathlib.Path(so).read_bytes()).hexdigest(),functions=items)
 (out/'target-sass-instructions.json').write_text(json.dumps(dict(scope='Static disassembly only; not dynamic executed instruction counts, achieved occupancy or bank-conflict counters.',libraries=sass),indent=2)+'\n')
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']},indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'counter-policy.txt').write_text(next(v for v in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in v)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
