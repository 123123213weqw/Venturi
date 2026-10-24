import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/batched-mtp-src-v5';out=root/'artifacts/venturi-batched-mtp-sdk-tests-v7';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 previous=root/'artifacts/venturi-batched-mtp-model-v2';gate=root/'artifacts/venturi-batched-mtp-final-v4'
 assert (gate/'exit-code.txt').read_text().strip()=='0'
 failed=root/'artifacts/venturi-batched-mtp-sdk-tests-v5';deadline=time.monotonic()+1200
 while not (failed/'exit-code.txt').exists():
  pid=json.loads((failed/'process.json').read_text())['pid'];c=pathlib.Path(f'/proc/{pid}/cmdline');assert c.exists() and str(failed/'worker.py').encode() in c.read_bytes();assert time.monotonic()<deadline;time.sleep(10)
 assert (failed/'exit-code.txt').read_text().strip()=='1'
 log=(failed/'full-tests.log').read_text();assert 'test_remote_launcher_quotes_prompt_and_paths' in log and '2 failed, 1990 passed, 448 skipped' in log
 (out/'recovery.json').write_text(json.dumps(dict(prior_failure_path=str(failed),prior_log_sha256=hashlib.sha256((failed/'full-tests.log').read_bytes()).hexdigest(),reason='Initial snapshot omitted the two launcher shell scripts. All 1990 other tests passed, including numerical/GPU cases. Fresh snapshot includes scripts; complete suite rerun before speed measurement. No code or numerical change.'),indent=2)+'\n')
 assert (previous/'exit-code.txt').read_text().strip()=='0'
 baseline=root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so'
 library=(previous/'library.txt').read_text().strip();build=json.loads(pathlib.Path(library).with_name('build.json').read_text());assert len(build['source_sha256'])==9
 assert hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest()=='f07bf75a975b963911aff5fefd5614dbf0b8b5b74e499f5525b6ad5d5317369a'
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'library.txt').write_text(library+'\n')
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=str(baseline),VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 env.update(VENTURI_Q8_SHARED_LIBRARY=library,VENTURI_Q8_FUSED_LIBRARY=library,VENTURI_Q8_TILED_SHARED_LIBRARY=library,VENTURI_SHARED_LIBRARY=library,VENTURI_SHARED_BASELINE_LIBRARY=str(root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'),VENTURI_WMMA_LIBRARY=library)
 env.update(VENTURI_GDN_COMPACT_LIBRARY=library,VENTURI_GDN_CONTROL_LIBRARY=str(baseline),VENTURI_MTP2_LIBRARY=library,VENTURI_MTP_PREFIX_LIBRARY=library,VENTURI_T3_LIBRARY=library,VENTURI_MTP_BATCH_LIBRARY=library,VENTURI_PREFILL_BUDGET_LIBRARY=library)
 run([py,'-m','pytest','-x','-q','tests/test_qwen_mtp_batch.py','--tb=short'],'mtp-tests.log',timeout=600)
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=600)
 q=previous/'qualification/report.json';assert json.loads(q.read_text())['completed']
 (out/'reused-trained-qualification.json').write_text(json.dumps(dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),scope='Same exact native SO and nine source dependencies. Only SDK redundant validation changed; every public metadata guard retained and covered by fresh CPU tests. Full GPU fixtures rerun. Original trained output/target/draft/private reset qualification reused, not a new quality campaign.'),indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
