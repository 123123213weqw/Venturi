import subprocess
worker=r'''import pathlib,subprocess,time,os,json,hashlib,traceback,signal,datetime
root=pathlib.Path('/home/wzu/codex-build/Venturi');out=root/'v100-mtp-diagnostics-v1';src=root/'v100-mtp-src-v4'
py='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def run(cmd,path,timeout=900):
 with path.open('w') as f:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned diagnostic exited {rc}: {cmd[0]}')
try:
 terminal=root/'v100-mtp-final-v4/exit-code.txt'
 for _ in range(150):
  if terminal.exists():break
  time.sleep(10)
 else:raise RuntimeError('Bounded 25 minute wait elapsed; no profiling started')
 assert terminal.read_text().strip()=='0','Final unprofiled comparison failed; no profiling started'
 assert json.loads((terminal.parent/'completed.json').read_text())['completed']
 library=pathlib.Path((terminal.parent/'library.txt').read_text().strip());digest=hashlib.sha256(library.read_bytes()).hexdigest()
 for mode,count in [('tp',1),('pp',8)]:
  d=out/f'{mode}-c{count}';d.mkdir()
  run(['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--cuda-event-trace=false','--stats=true','--output='+str(d/'trace'),py,'tools/benchmark_qwen_mtp.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--mtp-artifact',str(root/'models/qwen3.5-27b-mtp-q8.ninfer'),'--library',str(library),'--case',str(root/'v100-pp-final-v1/case.json'),'--mode',mode,'--output-directory',str(d/'profiled-run'),'--concurrency',str(count),'--new-tokens','8','--profile-nvtx'],d/'trace.log')
  r=json.loads((d/'profiled-run/report.json').read_text());assert r['completed'] and r['profiled'] and r['library_sha256']==digest
  run([py,'tools/summarize_native_parallel_trace.py',str(d/'trace.sqlite'),'--output',str(d/'cohort-attribution.json')],d/'attribution.log')
 run(['python3','tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',digest,'--trace','tp-c1='+str(out/'tp-c1/trace.sqlite'),'--trace','pp-c8='+str(out/'pp-c8/trace.sqlite'),'--output',str(out/'occupancy.json')],out/'occupancy.log')
 story=json.loads((root/'v100-pp-final-v1/case.json').read_text())
 run([py,'-m','venturi','qwen-native','generate',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--blocks-library',str(library),'--weight-mode','resident','--dtype','fp16','--mtp-artifact',str(root/'models/qwen3.5-27b-mtp-q8.ninfer'),'--tensor-parallel-device','cuda:1','--tensor-parallel-transport','host','--max-new-tokens','8','--max-tokens','23','--eos-token-ids','','--token-ids',','.join(map(str,story['prompt_token_ids']))],out/'cli-smoke.json')
 r=json.loads((out/'cli-smoke.json').read_text());assert r['generated_ids']==story['expected_generated_ids'][:8] and r['speculative_draft_tokens']==1
 run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-final.txt')
 (out/'policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 (out/'server-clock.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),utc_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds())),indent=2))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time(),library_sha256=digest),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
code=f'''import pathlib,subprocess,time,json
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-mtp-diagnostics-v1');p.mkdir();(p/'worker.py').write_text({worker!r})
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
