import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/dp4a-src-v1';out=root/'artifacts/venturi-dp4a-fixtures-v1';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 assert (root/'artifacts/venturi-native-t8-screen-v1/exit-code.txt').read_text().strip()=='0'
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(out/'build')],'build.log',timeout=180)
 library=(out/'build.log').read_text().splitlines()[-1];assert pathlib.Path(library).is_file();(out/'library.txt').write_text(library+'\n');env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-x','-q','tests/test_q8_dp4a.py','--tb=short'],'dp4a-tests.log',timeout=180)
 run([py,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_batch.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_mtp.py','tests/test_qwen_pipeline.py','--tb=short'],'default-regression-tests.log',timeout=180)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
