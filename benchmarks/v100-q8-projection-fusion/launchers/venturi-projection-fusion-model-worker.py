import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/projection-fusion-src-v2';out=root/'artifacts/venturi-projection-fusion-model-v2';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
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
 library=str(root/'artifacts/venturi-projection-fusion-v1/build/b08a62aa2ac57cc2/libventuri_blocks.so');assert hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest()=='6698f47bae3405279102890818782c992e8179aec966af8b5a08941160bb28b1'
 build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
 (out/'library.txt').write_text(library+'\n')
 rootgate=root/'artifacts/venturi-projection-fusion-v1';assert (rootgate/'exit-code.txt').read_text().strip()=='0'
 (out/'reused-fixture-gates.json').write_text(json.dumps(dict(path=str(rootgate),full_test_log_sha256=hashlib.sha256((rootgate/'full-tests.log').read_bytes()).hexdigest(),library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),scope='Identical nine native source hashes; 221 selected and 1517 full-suite passes reused, no new fixture-test execution.'),indent=2)+'\n')
 run([py,'-u','tools/benchmark_q8_fused.py',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'),'--library',library,'--baseline-library',str(baseline),'--output-directory',str(out/'micro')],'micro.log',timeout=300)
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';prior=root/'artifacts/venturi-tiled-staging-model-v3/qualification/report.json'
 run([py,'-u','tools/qualify_qwen_projection_variant.py',str(target),'--baseline-library',str(baseline),'--library',library,'--cases',str(cases),'--prior-report',str(prior),'--prior-report-sha256','dc3478d7dd188f03b0e8bcabfafcdc5a7ff6e8e9c15783b3d919b9ff11d514b9','--profiles','q8-residual','--reuse-prior-control','--reference-cases-sha256','fd5f82c3a4b4d22dfa05d577ba50c67019b47646440a54616e565d7bdf008d0b','--output-directory',str(out/'qualification')],'qualification.log',timeout=600)
 mtpref=root/'artifacts/venturi-shared-input-campaign-v3/mtp-qualification/report.json';assert hashlib.sha256(mtpref.read_bytes()).hexdigest()=='488346715a5c7c42fcc13901212e5c80b0b36b01dd3651bdb4378e9f058f0cd2'
 (out/'reused-draft-reference.json').write_text(json.dumps(dict(path=str(mtpref),sha256=hashlib.sha256(mtpref.read_bytes()).hexdigest(),scope='Only independent draft-block CPU reference reused; draft source and FP16 operators unchanged. Fresh trained target MTP committed-state/accept/replay qualification below covers changed same-input projection-group transport fences; unchanged FP16 draft arithmetic.'),indent=2)+'\n')
 run([py,'-u','tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--skip-block-reference','--output-directory',str(out/'mtp-qualification')],'mtp-qualification.log',timeout=600)
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
