import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/native-mtp2-src-v4';out=root/'artifacts/venturi-native-mtp2-prefix-model-v6';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 previous=root/'artifacts/venturi-native-mtp2-prefix-v5';assert (previous/'exit-code.txt').read_text().strip()=='0'
 library=(previous/'library.txt').read_text().strip();assert hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest()=='9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894'
 build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
 (out/'library.txt').write_text(library+'\n')
 (out/'reused-build.json').write_text(json.dumps(dict(path=str(previous),library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),source_sha256=build['source_sha256'],scope='Identical nine native sources; Python/CLI/qualification additions only, no rebuild.'),indent=2)+'\n')
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=str(baseline),VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 env.update(VENTURI_Q8_SHARED_LIBRARY=library,VENTURI_Q8_FUSED_LIBRARY=library,VENTURI_Q8_TILED_SHARED_LIBRARY=library,VENTURI_SHARED_LIBRARY=library,VENTURI_SHARED_BASELINE_LIBRARY=str(root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'),VENTURI_WMMA_LIBRARY=library)
 env.update(VENTURI_GDN_COMPACT_LIBRARY=library,VENTURI_GDN_CONTROL_LIBRARY=str(baseline),VENTURI_MTP2_LIBRARY=library,VENTURI_PREFILL_BUDGET_LIBRARY=library)
 (out/'reused-fixture-tests.json').write_text(json.dumps(dict(path=str(previous),sha256=hashlib.sha256((previous/'full-tests.log').read_bytes()).hexdigest(),scope='Same exact source and library. Prefix, MTP2 and complete fixture suite already passed: 1768 passed, 448 skipped.'),indent=2)+'\n')
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';prior=root/'artifacts/venturi-gdn-compact-model-v2/qualification/report.json'
 run([py,'-u','tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--draft-tokens','2','--recovery','prefix','--skip-block-reference','--prior-target-report',str(prior),'--prior-target-report-sha256','a281b7b4fc5528d0c46eb1f2506f1f59f75c6f4c806800e8ffd765e051d4e275','--output-directory',str(out/'qualification')],'qualification.log',timeout=900)
 q=json.loads((out/'qualification/report.json').read_text());assert q['completed'] and q['recovery']=='prefix'
 assert all(c['mtp_stats']['replayed_input_tokens']==0 and c['mtp_stats']['target_recovery']=='prefix-trace' and c['mtp_stats']['prefix_trace_extra_primary_bytes']==c['mtp_stats']['primary_target_backup_bytes'] for c in q['cases'])
 (out/'reused-target-reference.json').write_text(json.dumps(dict(path=str(prior),sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),scope='Unchanged target arithmetic and fixed CPU/teacher-vector gates reused through exact pinned four-trajectory final-state checks. Fresh ordinary/MTP1/MTP2 target/draft comparisons plus three-step trained recursive draft reference are executed.'),indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
