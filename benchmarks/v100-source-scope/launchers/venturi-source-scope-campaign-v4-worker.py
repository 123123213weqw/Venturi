import pathlib,subprocess,json,os,time,traceback,signal,urllib.request,hashlib,datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-source-scope-campaign-v4';src=root/'repos/codex-build/Venturi/source-scope-src-v3';prior=root/'artifacts/venturi-source-scope-model-v3'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';port=18953
bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',NCCL_P2P_DISABLE='1',LD_LIBRARY_PATH=str(bindir)+':/usr/local/cuda/lib64:'+str(root/'repos/codex-build/.venvs/b2-v100/lib/python3.10/site-packages/nvidia/nccl/lib'))
report=dict(completed=False,scope='Explicit residual-A8 draft vs original FP16 draft; same native library/SDK, qualified column-MLP target, fixed Q8 weights/FP16 KV/work/policies. Complete HTTP cohort includes reset/admission/replies/client archival; native decode excludes those. Changed draft arithmetic separately qualified; not old-profile byte parity or BF16 corpus quality.',experiments=[],physical_gpus=[1,0],utc_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds()),achieved_hardware_counters=None,path_promotions={},screen_ratios={})
def save():(out/'campaign-report.json').write_text(json.dumps(report,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stop(p):
 if p and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run(cmd,path,timeout=600):
 (path.with_suffix(path.suffix+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with path.open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:stop(p);raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {path}')
def experiment(label,engine,mode,depth,counts,repeats,recovery="replay"):
 directory=out/label;directory.mkdir();server=None;samplers=[];logs=[];item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,mtp_recovery=recovery,projection_rows=selected_rows,projection_profile=selected_profile if engine=='venturi' else None,mtp_draft_projection_profile=selected_draft if engine=='venturi' and depth else None,library_sha256=digest(library) if engine=='venturi' else None,candidate_build=(selected_draft=='q8-residual') if engine=='venturi' else None,completed=False)
 try:
  for gpu in [1,0]:
   f=(directory/f'gpu-{gpu}-samples.csv').open('wb');logs.append(f);samplers.append(subprocess.Popen(['nvidia-smi','-i',str(gpu),'--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits','-lms','100'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
  if engine=='llama':
   cmd=[str(bindir/'llama-server'),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-ngl','all','--split-mode',mode,'--tensor-split','1,1','--fit','off','-c','2048','--parallel','8','-b','256','-ub','256','-t','4','-tb','4','-fa','on','-ctk','f16','-ctv','f16','--backend-sampling','--spec-type','draft-mtp' if depth else 'none']
   if depth:cmd+=['--spec-draft-n-max',str(depth),'--spec-draft-backend-sampling']
  else:
   cmd=[py,'tools/serve_native_benchmark.py',str(target),'--library',str(library),'--mode','tp','--port',str(port),'--slots','8','--capacity','256','--ready-file',str(directory/'ready.json')]
   cmd+=['--projection-profile',selected_profile,'--projection-rows',str(selected_rows)]
   if depth:cmd+=['--mtp-draft-projection-profile',selected_draft,'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--mtp-depth',str(depth),'--mtp-recovery',recovery]
   if mode=='batch':cmd+=['--batch-target','--batch-window-ms','25']
   if mode=='mtp-batch':cmd+=['--batch-mtp','--batch-window-ms','25']
  item['command']=cmd;(directory/'command.json').write_text(json.dumps(cmd,indent=2));f=(directory/'server.log').open('wb');logs.append(f);server=subprocess.Popen(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);item['pid']=server.pid
  for attempt in range(180):
   if server.poll() is not None:raise RuntimeError(f'Server exited during startup: {server.returncode}')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:item['health']=json.load(response);break
   except Exception:time.sleep(1)
  else:raise RuntimeError('Bounded health wait elapsed')
  assert item['health']['mtp_draft_projection_profile']==(selected_draft if depth else None)
  assert item['health']['source_validation_policy'].startswith('resident-forward')==(selected_scope==1)
  item['source_scope_mode']=selected_scope
  (directory/'loaded-maps.txt').write_text('\n'.join(s for s in pathlib.Path(f'/proc/{server.pid}/maps').read_text().splitlines() if '.so' in s))
  cmd=[py,'-u','tools/benchmark_completion_api.py','--url',f'http://127.0.0.1:{port}','--server-pid',str(server.pid),'--case',str(case),'--concurrency',counts,'--repeats',str(repeats),'--output-directory',str(directory/'timed'),'--expect-pinned-ids']
  run(cmd,directory/'benchmark.log',timeout=600);measured=json.loads((directory/'timed/report.json').read_text());assert measured['completed']
  if engine=='venturi' and mode in ['batch','mtp-batch']:
   for w in measured['waves']:
    assert {r['response']['batch_request_count'] for r in w['requests']}=={w['request_count']},'Admission split a fixed cohort'
    assert len({r['response']['batch_cohort_id'] for r in w['requests']})==1
  item.update(completed=True,summary=measured['summary'],stable_outputs=measured['all_outputs_equal_first_wave'])
 except BaseException:item['failure']=traceback.format_exc();(directory/'failure.txt').write_text(item['failure']);raise
 finally:
  stop(server)
  for p in samplers:stop(p)
  for f in logs:f.close()
  report['experiments'].append(item);save()
 print(json.dumps(item),flush=True);return item

import statistics
try:
 deadline=time.monotonic()+3600
 while not (prior/'exit-code.txt').exists():
  pid=json.loads((prior/'process.json').read_text())['pid'];p=pathlib.Path(f'/proc/{pid}/cmdline');assert p.exists() and str(prior/'worker.py').encode() in p.read_bytes(),'Prerequisite missing without terminal marker'
  assert time.monotonic()<deadline,'Verified live prerequisite bounded wait elapsed';time.sleep(3)
 assert (prior/'exit-code.txt').read_text().strip()=='0','Model qualification failed'
 qualified=json.loads((prior/'completed.json').read_text());assert qualified['model_qualified'] and qualified['fresh_trained_exact_scalar_and_cohort_states']
 actual={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==json.loads((prior/'executed-source.json').read_text())
 fixtures=root/'artifacts/venturi-source-scope-fixtures-v2';libs={0:pathlib.Path((fixtures/'default-library.txt').read_text().strip()),1:pathlib.Path((fixtures/'scoped-library.txt').read_text().strip())}
 for enabled,lib in libs.items():
  build=json.loads(lib.with_name('build.json').read_text());assert build['resident_source_scope']==bool(enabled) and digest(lib)==build['library_sha256']
  for rel,h in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==h
 report.update(scope='Lexical resident source validation only: forced entry/exit and all streaming/unscoped guards preserved; no arithmetic change. Complete HTTP cohort includes reset/admission/queue/reply/client archival. Three paired C1/C8 screen trials with reversed order in trial2. No strace/nsys in this worker.',model_qualified=True,source_sha256=actual,library_sha256_by_scope={str(k):digest(v) for k,v in libs.items()},path_promotions={},screen_ratios={},fresh_external_campaign_run=False,external_goal_complete=False)
 selected_profile='q8-residual-column-mlp';selected_draft='q8-residual'
 def summary(e,n):return next(r for r in e['summary'] if r['request_count']==n)
 def combined(items,n):
  rows=[summary(e,n) for e in items];wall=statistics.mean(r['mean_cohort_wall_seconds'] for r in rows);result=dict(http_tps=32*n/wall)
  times=[t for r in rows for t in r.get('engine_reported_timings',[])]
  if times:result['single_post_first_tps']=31/statistics.mean(t['native_generation_seconds']-t['native_time_to_first_token_seconds'] for t in times)
  return result
 paths=[('mtp2','scalar',2,'prefix',32),('mtp-batch','mtp-batch',1,'replay',32),('ordinary256','batch',0,'replay',256)]
 for name,mode,depth,recovery,rows in paths:
  selected_rows=rows;arms={0:[],1:[]}
  for trial in range(3):
   for selected_scope in ([0,1] if trial!=1 else [1,0]):
    library=libs[selected_scope];item=experiment('screen-'+name+'-trial'+str(trial+1)+'-scope'+str(selected_scope),'venturi',mode,depth,'1,8',1,recovery);arms[selected_scope].append(item)
  c0,c1=combined(arms[0],1),combined(arms[1],1);e0,e1=combined(arms[0],8),combined(arms[1],8)
  http=c1['http_tps']/c0['http_tps'];eight=e1['http_tps']/e0['http_tps'];decode=c1.get('single_post_first_tps',0)/c0.get('single_post_first_tps',1)
  hi=sum(summary(b,1)['aggregate_complete_request_output_tps']>summary(a,1)['aggregate_complete_request_output_tps'] for a,b in zip(arms[0],arms[1]))
  di=sum(summary(b,1).get('native_single_post_first_tps',0)>summary(a,1).get('native_single_post_first_tps',0) for a,b in zip(arms[0],arms[1]))
  gate=((http>1.01 and hi>=2) or (decode>1.01 and di>=2)) and http>=0.99 and eight>=0.98
  report['screen_ratios'][name]=dict(c1_http=http,c1_decode=decode,c8_http=eight,improving_http_trials=hi,improving_decode_trials=di)
  report['path_promotions'][name]=gate;save()
 for name,mode,depth,recovery,rows in paths:
  if not report['path_promotions'][name]:continue
  selected_rows=rows
  for selected_scope in [0,1]:
   library=libs[selected_scope];experiment('repeat-'+name+'-scope'+str(selected_scope),'venturi',mode,depth,'1,2,4,8',3,recovery)
 report['promotion_rule']='Three paired screens; >1% C1 decode or C1 HTTP with >=2 improving paired trials in that metric, no >1% C1 HTTP or >2% C8 HTTP regression; promoted paths require 3-repeat1/2/4/8 matrix. Full external goal unchanged.'
 report['retained_stronger_controls']=[dict(path=str(root/'artifacts/venturi-mtp-draft-residual-campaign-v3/campaign-report.json'),sha256=digest(root/'artifacts/venturi-mtp-draft-residual-campaign-v3/campaign-report.json'))]
 report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 try:run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 except BaseException:(out/'cleanup-observation-failure.txt').write_text(traceback.format_exc())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
