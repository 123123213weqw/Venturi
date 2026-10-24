import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/tiled-staging-src-v1';out=root/'artifacts/venturi-tiled-staging-v1';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 baseline=root/'artifacts/venturi-shared-activation-model-v2/build/43feb26c8cde610e/libventuri_blocks.so';assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='1aba1423dc511b7cb17f03c0a4b71948f5fdd4f836e0dcfe9e0190b15bddbee7'
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(out/'build')],'build.log',timeout=240)
 library=(out/'build.log').read_text().splitlines()[-1];assert pathlib.Path(library).is_file();(out/'library.txt').write_text(library+'\n')
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=str(baseline),VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 env.update(VENTURI_Q8_SHARED_LIBRARY=library,VENTURI_Q8_TILED_SHARED_LIBRARY=library,VENTURI_WMMA_LIBRARY=library)
 run([py,'-m','pytest','-x','-q','tests/test_q8_tiled_shared.py','tests/test_q8_shared.py','tests/test_q8_dp4a_variant.py','tests/test_q8_dp4a.py','--tb=short'],'profile-tests.log',timeout=240)
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=300)
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'
 run([py,'-u','tools/benchmark_q8_tiled_shared.py',str(target),'--baseline-library',str(baseline),'--library',library,'--output-directory',str(out/'projection')],'projection.log',timeout=600)
 nsys='/usr/local/bin/nsys'
 run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(out/'projection-trace'),py,'tools/profile_q8_tiled_shared.py',str(target),'--library',library,'--baseline-library',str(baseline),'--output',str(out/'projection-scope.json')],'projection-profile.log',timeout=240)
 run([nsys,'export','--type','sqlite','--output',str(out/'projection-trace.sqlite'),str(out/'projection-trace.nsys-rep')],'projection-export.log',timeout=120)
 run([py,'tools/summarize_q8_projection_trace.py',str(out/'projection-trace.sqlite'),'--output',str(out/'projection-attribution.json')],'projection-attribution.log',timeout=90)
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),'--trace','projection='+str(out/'projection-trace.sqlite'),'--output',str(out/'resources.json')],'resources.log',timeout=120)
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']},indent=2)+'\n')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
