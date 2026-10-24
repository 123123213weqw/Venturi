import pathlib,subprocess,time,os,json,hashlib,traceback,signal
ROOT=pathlib.Path('/home/wzu/codex-build/Venturi');OUT=ROOT/'v100-occupancy-diagnostics-v1';SRC=ROOT/'v100-occupancy-src-v1'
PY='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(ROOT/'v100-tp-diagnostics-v2/python-deps')+':'+str(SRC/'src')+':'+str(SRC/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def run(cmd,path,timeout=900):
 with path.open('w') as f:
  p=subprocess.Popen(cmd,cwd=SRC,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned diagnostic exited {rc}: {cmd[0]}')
try:
 terminal=ROOT/'v100-occupancy-prefill-v2/exit-code.txt'
 for _ in range(120):
  if terminal.exists():break
  time.sleep(10)
 else:raise RuntimeError('Bounded 20 minute wait elapsed; no profiling started')
 assert terminal.read_text().strip()=='0','Final unprofiled benchmark failed; no profiling started'
 assert json.loads((terminal.parent/'completed.json').read_text())['completed']
 library=pathlib.Path((terminal.parent/'library.txt').read_text().strip())
 digest=hashlib.sha256(library.read_bytes()).hexdigest()
 for mode,count in [('tp',1),('pp',8)]:
  d=OUT/f'{mode}-c{count}';d.mkdir()
  run(['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--cuda-event-trace=false','--stats=true','--output='+str(d/'trace'),PY,'tools/benchmark_qwen_native_parallel.py',str(ROOT/'models/qwen3.5-27b-text-q8.ninfer'),'--library',str(library),'--case',str(ROOT/'v100-pp-final-v1/case.json'),'--mode',mode,'--output-directory',str(d/'profiled-run'),'--concurrency',str(count),'--repeats','1','--new-tokens','8','--split-layer','32','--profile-nvtx'],d/'trace.log')
  report=json.loads((d/'profiled-run/report.json').read_text());assert report['completed'] and report['profiled'] and report['library_sha256']==digest
  run([PY,'tools/summarize_native_parallel_trace.py',str(d/'trace.sqlite'),'--output',str(d/'cohort-attribution.json')],d/'attribution.log')
 run(['python3','tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',digest,'--trace','tp-c1='+str(OUT/'tp-c1/trace.sqlite'),'--trace','pp-c8='+str(OUT/'pp-c8/trace.sqlite'),'--output',str(OUT/'occupancy.json')],OUT/'occupancy.log')
 run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],OUT/'gpu-final.txt')
 (OUT/'policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 (OUT/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time(),library_sha256=digest),indent=2));rc=0
except BaseException:
 (OUT/'failure.txt').write_text(traceback.format_exc());rc=1
(OUT/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
