import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/layer-graphs-src-v3';out=root/'artifacts/venturi-layer-graphs-final-v6';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run(cmd,name,timeout=600):
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/name).open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:stop(p);raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
try:
 gate=root/'artifacts/venturi-layer-graphs-campaign-v5';deadline=time.monotonic()+3600
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];cmdline=pathlib.Path(f'/proc/{pid}/cmdline');assert cmdline.exists() and str(gate/'worker.py').encode() in cmdline.read_bytes(),'Pinned predecessor absent';assert time.monotonic()<deadline,'Bounded predecessor wait elapsed';time.sleep(10)
 assert (gate/'exit-code.txt').read_text().strip()=='0';campaign=json.loads((gate/'campaign-report.json').read_text());assert campaign['completed']
 previous=root/'artifacts/venturi-layer-graphs-fixtures-v3';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text());baseline=str(root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so')
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'reused-fixture-tests.json').write_text(json.dumps(dict(path=str(previous),sha256=hashlib.sha256((previous/'full-tests.log').read_bytes()).hexdigest(),scope='Exact native library and nine dependency hashes; full fixture suite reused. Whole-call profiles remain separate from unprofiled matrices.'),indent=2)+'\n')
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';nsys='/usr/local/bin/nsys';case=json.loads((root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json').read_text());(out/'profile-case.json').write_text(json.dumps(case,indent=2)+'\n');traces=[]
 if campaign['operator_promoted']:
  for name,cmd in [
   ('mtp2-eager-c1',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','1']),
   ('mtp2-graph-c1',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','1','--target-layer-graph']),
   ('mtp2-graph-c8',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','8','--target-layer-graph'])]:
   prefix=out/name;cmd+=['--output',str(out/(name+'-metadata.json'))]
   run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--cuda-graph-trace=node','--force-overwrite=false','--output',str(prefix),*cmd],name+'-profile.log',timeout=600)
   run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=240)
   run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=240)
   traces+=['--trace',name+'='+str(prefix)+'.sqlite']
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
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']},indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
