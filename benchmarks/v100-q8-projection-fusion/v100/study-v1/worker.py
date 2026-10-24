import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/projection-fusion-src-v1';out=root/'artifacts/venturi-projection-fusion-v1';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 baseline=root/'artifacts/venturi-tiled-staging-model-v3/build/398e4f4a1dd037ea/libventuri_blocks.so';assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='a6599e636cc0c018261310de7d83e1e8dd41d6edcc635f9161402a89f7775a3f'
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(out/'build')],'build.log',timeout=240)
 library=(out/'build.log').read_text().splitlines()[-1];assert pathlib.Path(library).is_file();(out/'library.txt').write_text(library+'\n')
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=str(baseline),VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 env.update(VENTURI_Q8_SHARED_LIBRARY=library,VENTURI_Q8_FUSED_LIBRARY=library,VENTURI_Q8_TILED_SHARED_LIBRARY=library,VENTURI_SHARED_LIBRARY=library,VENTURI_SHARED_BASELINE_LIBRARY=str(root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'),VENTURI_WMMA_LIBRARY=library)
 run([py,'-m','pytest','-x','-q','tests/test_q8_fused.py','tests/test_q8_shared.py','tests/test_q8_dp4a_variant.py','tests/test_q8_dp4a.py','--tb=short'],'profile-tests.log',timeout=240)
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=300)
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
