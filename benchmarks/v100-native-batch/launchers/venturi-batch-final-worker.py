import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/native-batch-src-v3';out=root/'artifacts/venturi-native-batch-final-v5';campaign=root/'artifacts/venturi-native-batch-v4';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_BATCH_TEST='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
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
 for _ in range(180):
  if (campaign/'exit-code.txt').exists():break
  time.sleep(5)
 else:raise RuntimeError('Finite campaign wait elapsed; no new GPU tests started')
 library=(root/'artifacts/venturi-native-batch-v2/library.txt').read_text().strip();env['VENTURI_SM70_LIBRARY']=library
 run([py,'-m','pytest','-q','--tb=short'],'tests.log',timeout=300)
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],'remaining-gpu-processes.txt')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 proof=json.loads((campaign/'campaign-report.json').read_text());checks={}
 for rel,sha in proof['source_hashes'].items():
  actual=hashlib.sha256((src/rel).read_bytes()).hexdigest();assert actual==sha;checks[rel]=actual
 (out/'executed-source.json').write_text(json.dumps(dict(source_sha256=checks,library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),campaign_exit_code=(campaign/'exit-code.txt').read_text().strip(),exact_source_unchanged=True),indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
