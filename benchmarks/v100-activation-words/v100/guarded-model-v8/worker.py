import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/activation-words-src-v4';out=root/'artifacts/venturi-activation-words-guarded-model-v8';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 gate=root/'artifacts/venturi-activation-words-guarded-study-v7';deadline=time.monotonic()+1800
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];c=pathlib.Path(f'/proc/{pid}/cmdline');assert c.exists() and str(gate/'worker.py').encode() in c.read_bytes(),'Pinned predecessor absent';assert time.monotonic()<deadline,'Observation deadline; revalidate original handle, never relaunch';time.sleep(10)
 assert (gate/'exit-code.txt').read_text().strip()=='0'
 op=json.loads((gate/'operators/report.json').read_text());assert op['completed'] and len(op['cases'])==33 and all(c['all_output_bytes_equal'] and len(c['waves'])==12 and all(w['calls']==9 for w in c['waves']) for c in op['cases'])
 import statistics
 arm='candidate-guarded-pairs';assert op['candidate_arm']==arm
 ratios=[c['median_gpu_event_ms']['control-aos']/c['median_gpu_event_ms'][arm] for c in op['cases'] if c['label'].startswith('fused-') and c['tokens']==3]
 neutral=[c['median_gpu_event_ms']['control-aos']/c['median_gpu_event_ms'][arm] for c in op['cases'] if c['label'].startswith('fused-') and c['tokens']==8]
 promoted=statistics.median(ratios)>=1.02 and min(neutral)>=.98
 (out/'operator-promotion.json').write_text(json.dumps(dict(promoted=promoted,median_fused_t3_ratio=statistics.median(ratios),minimum_unchanged_fused_t8_ratio=min(neutral),rule='Original declared gate: median fused T3 control/candidate >=1.02 and minimum unchanged fused T8 >=.98. All original-control/candidate arms retained; no model win inferred.'),indent=2)+'\n')
 library=(gate/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'reused-fixture-tests.json').write_text(json.dumps(dict(path=str(gate),sha256=hashlib.sha256((gate/'full-tests.log').read_bytes()).hexdigest(),scope='Identical native SO, all nine dependency hashes and Python snapshot; full fixture suite reused. No second build or repeated unchanged fixture campaign.'),indent=2)+'\n')
 if promoted:
  target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';prior=root/'artifacts/venturi-gdn-compact-model-v2/qualification/report.json';qualified=root/'artifacts/venturi-gdn-compact-model-v2/build/0849ea2437e65c8a/libventuri_blocks.so'
  run([py,'-u','tools/qualify_qwen_projection_variant.py',str(target),'--baseline-library',str(qualified),'--library',library,'--projection-rows','256','--cases',str(cases),'--prior-report',str(prior),'--prior-report-sha256','a281b7b4fc5528d0c46eb1f2506f1f59f75c6f4c806800e8ffd765e051d4e275','--profiles','q8-residual','--reuse-prior-control','--reference-cases-sha256','fd5f82c3a4b4d22dfa05d577ba50c67019b47646440a54616e565d7bdf008d0b','--output-directory',str(out/'qualification')],'qualification.log',timeout=600)
  run([py,'-u','tools/qualify_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--cases',str(cases),'--prior-target-report',str(prior),'--prior-target-report-sha256','a281b7b4fc5528d0c46eb1f2506f1f59f75c6f4c806800e8ffd765e051d4e275','--output-directory',str(out/'mtp-batch-qualification')],'mtp-batch-qualification.log',timeout=900)
  run([py,'-u','tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--draft-tokens','2','--recovery','prefix','--skip-block-reference','--prior-target-report',str(prior),'--prior-target-report-sha256','a281b7b4fc5528d0c46eb1f2506f1f59f75c6f4c806800e8ffd765e051d4e275','--output-directory',str(out/'mtp2-qualification')],'mtp2-qualification.log',timeout=900)
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,operator_promoted=promoted,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
