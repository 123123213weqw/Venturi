import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/strided-src-v2';out=root/'artifacts/venturi-strided-fixtures-v3';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 baseline=root/'artifacts/venturi-wmma-guarded-v4/build/8537a2e208c36c23/libventuri_blocks.so';assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='ccf03d8caa89ced089b80cd5b8eb49cb494e53baf855c78c99e9c8f5ed9627c5'
 library=(root/'artifacts/venturi-strided-fixtures-v1/library.txt').read_text().strip();build=json.loads(pathlib.Path(library).with_name('build.json').read_text());assert pathlib.Path(library).is_file()
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'library.txt').write_text(library+'\n');(out/'reused-build.json').write_text(json.dumps(build,indent=2)+'\n')
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=str(baseline),VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-x','-q','tests/test_q8_dp4a_strided.py','tests/test_q8_dp4a_variant.py','tests/test_q8_dp4a.py','--tb=short'],'profile-tests.log',timeout=240)
 run([py,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_batch.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_mtp.py','tests/test_qwen_pipeline.py','--tb=short'],'default-regression-tests.log',timeout=240)
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';prior=root/'artifacts/venturi-wmma-guarded-v4/qualification/report.json'
 run([py,'-u','tools/qualify_qwen_projection_variant.py',str(target),'--baseline-library',str(baseline),'--library',library,'--cases',str(cases),'--prior-report',str(prior),'--prior-report-sha256','857df38e8b8e8db5222f584469fcdddb8d85aa0049557e57da699871184ee5f1','--output-directory',str(out/'qualification')],'qualification.log',timeout=600)
 mtpref=root/'artifacts/venturi-shared-input-campaign-v3/mtp-qualification/report.json';assert hashlib.sha256(mtpref.read_bytes()).hexdigest()=='488346715a5c7c42fcc13901212e5c80b0b36b01dd3651bdb4378e9f058f0cd2'
 (out/'reused-draft-reference.json').write_text(json.dumps(dict(path=str(mtpref),sha256=hashlib.sha256(mtpref.read_bytes()).hexdigest(),scope='Only independent draft-block CPU reference reused; draft source and FP16 operators unchanged. Fresh trained target MTP committed-state/accept/replay qualification below covers changed T2 target pitched stores.'),indent=2)+'\n')
 run([py,'-u','tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--skip-block-reference','--output-directory',str(out/'mtp-qualification')],'mtp-qualification.log',timeout=600)
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
