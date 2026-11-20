import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/batched-mtp-src-v3';out=root/'artifacts/venturi-batched-mtp-final-v4';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 gate=root/'artifacts/venturi-batched-mtp-campaign-v3';deadline=time.monotonic()+1500
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];cmdline=pathlib.Path(f'/proc/{pid}/cmdline');assert cmdline.exists() and str(gate/'worker.py').encode() in cmdline.read_bytes(),'Pinned predecessor absent';assert time.monotonic()<deadline,'Bounded predecessor wait elapsed';time.sleep(10)
 assert (gate/'exit-code.txt').read_text().strip()=='0';campaign=json.loads((gate/'campaign-report.json').read_text());assert campaign['completed']
 previous=root/'artifacts/venturi-batched-mtp-model-v2';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text());baseline=str(root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so')
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'reused-fixture-tests.json').write_text(json.dumps(dict(path=str(previous),sha256=hashlib.sha256((previous/'full-tests.log').read_bytes()).hexdigest(),scope='Same exact library and nine sources: Identical qualified nine-source candidate and complete fixture suite reused. Fresh full32-output MTP1 target-batch profiles exercise real native verification, one C-call range per logical cohort.'),indent=2)+'\n')
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';nsys='/usr/local/bin/nsys';case=json.loads((root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json').read_text());(out/'profile-case.json').write_text(json.dumps(case,indent=2)+'\n');traces=[]
 for count in [1,8]:
  name=f'mtp-batch-c{count}';prefix=out/name
  run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--count',str(count),'--output',str(out/(name+'-metadata.json'))],name+'-profile.log',timeout=300)
  run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=120)
  run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=90)
  traces+=['--trace',name+'='+str(prefix)+'.sqlite']
 control=root/'artifacts/venturi-native-mtp2-prefix-v5/build/48a678383e862400/libventuri_blocks.so';assert hashlib.sha256(control.read_bytes()).hexdigest()=='9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894'
 name='ordinary256-control-c8';prefix=out/name
 run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_qwen_batch.py',str(target),'--library',str(control),'--case',str(out/'profile-case.json'),'--mode','batch','--projection-profile','q8-residual','--projection-rows','256','--count','8','--output',str(out/(name+'-metadata.json'))],name+'-profile.log',timeout=300)
 run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=120)
 run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=90)
 run([py,'tools/inspect_cuda_occupancy.py','--library',str(control),'--expected-sha256',hashlib.sha256(control.read_bytes()).hexdigest(),'--trace',name+'='+str(prefix)+'.sqlite','--output',str(out/'control-resources.json')],'control-resources.log',timeout=180)
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),*traces,'--output',str(out/'resources.json')],'resources.log',timeout=180)
 old=root/'artifacts/venturi-native-mtp2-full-profile-v14';reused={}
 for name in ['mtp2-prefix-full-c1-metadata.json','mtp2-prefix-full-c8-metadata.json','mtp2-prefix-full-c1-attribution.json','mtp2-prefix-full-c8-attribution.json','resources.json','remote-retained-files.json','executed-source.json']:
  p=old/name;reused[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (out/'reused-control-profiles.json').write_text(json.dumps(reused,indent=2)+'\n')
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']},indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
