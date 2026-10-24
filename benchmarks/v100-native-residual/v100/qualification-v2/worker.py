import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/model-profile-src-v3';out=root/'artifacts/venturi-model-profile-qualification-v2';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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

 previous=root/'artifacts/venturi-model-profile-fixtures-v1';assert (previous/'exit-code.txt').read_text().strip()=='0';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
 env.update(VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_MTP_TEST='1',VENTURI_TP_TEST='1',VENTURI_BATCH_TEST='1',VENTURI_PP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-x','-q','tests/test_qwen_projection_profile.py','--tb=short'],'profile-tests.log',timeout=180)
 run([py,'-u','tools/qualify_qwen_projection_profile.py',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'),'--library',library,'--cases',str(root/'repos/codex-build/Venturi/venturi-mtp-cases.json'),'--cpu-reference-directory',str(root/'repos/codex-build/Venturi/greedy-validation-v3/full-model-run1'),'--output-directory',str(out/'qualification')],'qualification.log',timeout=480)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
