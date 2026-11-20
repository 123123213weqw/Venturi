from pathlib import Path
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
old=Path('benchmarks/v100-mtp-draft-residual/v100/campaign-v3/worker.py').read_text();source=old[:old.index('\n\n\n\ntry:\n')]
source=source.replace('venturi-mtp-draft-residual-campaign-v3','venturi-source-scope-campaign-v4').replace('mtp-draft-residual-src-v1','source-scope-src-v3').replace('venturi-mtp-draft-residual-model-v2','venturi-source-scope-model-v3')
source=source.replace("scope='Explicit residual-A8 draft vs original FP16 draft; same native library/SDK, qualified column-MLP target, fixed Q8 weights/FP16 KV/work/policies. Complete HTTP cohort includes reset/admission/queue/reply/client archival.'","scope='Experimental lexical resident source validation vs default per-weight checks; same SDK/native arithmetic, qualified column-MLP target/residual draft, fixed Q8 weights/FP16 KV/work/policies. Complete HTTP cohort includes reset/admission/queue/reply/client archival.'")
source=source.replace("  assert item['health']['mtp_draft_projection_profile']==selected_draft", "  assert item['health']['mtp_draft_projection_profile']==(selected_draft if depth else None)\n  assert item['health']['source_validation_policy'].startswith('resident-forward')==(selected_scope==1)\n  item['source_scope_mode']=selected_scope")
source+='''
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
'''
ns['dispatch']('venturi-source-scope-campaign-v4',source)
