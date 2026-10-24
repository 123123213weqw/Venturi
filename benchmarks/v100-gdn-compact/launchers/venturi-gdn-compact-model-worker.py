import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/gdn-compact-src-v2';out=root/'artifacts/venturi-gdn-compact-model-v2';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 baseline=root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so';assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='698452e2b2a39f3a5bb4d08d58e37a3330857f6444cc462f96147f53a7ab5e3d'
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(out/'build')],'build.log',timeout=240)
 library=(out/'build.log').read_text().splitlines()[-1];assert pathlib.Path(library).is_file();(out/'library.txt').write_text(library+'\n')
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=str(baseline),VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 env.update(VENTURI_Q8_SHARED_LIBRARY=library,VENTURI_Q8_FUSED_LIBRARY=library,VENTURI_Q8_TILED_SHARED_LIBRARY=library,VENTURI_SHARED_LIBRARY=library,VENTURI_SHARED_BASELINE_LIBRARY=str(root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'),VENTURI_WMMA_LIBRARY=library)
 env.update(VENTURI_GDN_COMPACT_LIBRARY=library,VENTURI_GDN_CONTROL_LIBRARY=str(baseline))
 run([py,'-m','pytest','-x','-q','tests/test_gdn_compact.py','--tb=short'],'operator-tests.log',timeout=240)
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=300)
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';prior=root/'artifacts/venturi-projection-fusion-model-v3/qualification/report.json'
 run([py,'-u','tools/qualify_qwen_projection_variant.py',str(target),'--baseline-library',str(baseline),'--library',library,'--cases',str(cases),'--prior-report',str(prior),'--prior-report-sha256','f275b50a79a69ad472b08b0d89e61882174d2aef181f5f6fce384ed7df4ec645','--profiles','q8-residual','--reuse-prior-control','--reference-cases-sha256','fd5f82c3a4b4d22dfa05d577ba50c67019b47646440a54616e565d7bdf008d0b','--output-directory',str(out/'qualification')],'qualification.log',timeout=600)
 mtpref=root/'artifacts/venturi-shared-input-campaign-v3/mtp-qualification/report.json';assert hashlib.sha256(mtpref.read_bytes()).hexdigest()=='488346715a5c7c42fcc13901212e5c80b0b36b01dd3651bdb4378e9f058f0cd2'
 (out/'reused-draft-reference.json').write_text(json.dumps(dict(path=str(mtpref),sha256=hashlib.sha256(mtpref.read_bytes()).hexdigest(),scope='Only independent draft-block CPU reference reused; draft source and FP16 operators unchanged. Fresh trained target MTP committed-state/accept/replay qualification below covers changed residual-target compact GDN geometry; unchanged default FP16 draft/head arithmetic.'),indent=2)+'\n')
 run([py,'-u','tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--skip-block-reference','--output-directory',str(out/'mtp-qualification')],'mtp-qualification.log',timeout=600)
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
