from pathlib import Path
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
s=Path('/tmp/venturi-layer-graphs-campaign-v5-worker.py').read_text()
start=s.index('candidate_library=library;');head=s[:start]
head=head.replace('venturi-layer-graphs-campaign-v5','venturi-native-column-mlp-campaign-v3').replace('layer-graphs-src-v3','native-column-mlp-src-v2').replace('venturi-layer-graphs-model-v4','venturi-native-column-mlp-model-v2')
head=head.replace("   if library==candidate_library and enable_graph:cmd+=['--target-layer-graph']\n",'')
body=r'''candidate_library=library
legacy=root/'artifacts/venturi-native-mtp2-prefix-v5/build/48a678383e862400/libventuri_blocks.so'
cohort_control=root/'artifacts/venturi-batched-mtp-model-v2/build/e94e01699eaad5b0/libventuri_blocks.so'
selected_profile='q8-residual';selected_rows=32
try:
 assert digest(legacy)=='9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894'
 assert digest(cohort_control)=='f07bf75a975b963911aff5fefd5614dbf0b8b5b74e499f5525b6ad5d5317369a'
 actual={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert actual==json.loads((prior/'pre-execution-source.json').read_text())
 model=json.loads((prior/'completed.json').read_text());assert model['model_qualified']
 q=json.loads((prior/'qualification/report.json').read_text());assert q['completed'] and q['profiles']['q8-residual-column-mlp']['candidate']['passed']
 q=json.loads((prior/'mtp-batch-qualification/report.json').read_text());assert q['completed'] and q['selective_reset_untouched_state_exact']
 q=json.loads((prior/'mtp2-qualification/report.json').read_text());assert q['completed'] and len(q['cases'])==4 and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['committed_draft_state_equal_mtp1'] for c in q['cases'])
 build=json.loads(candidate_library.with_name('build.json').read_text());assert len(build['source_sha256'])==10
 for rel,h in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==h
 report.update(scope='Native MLP-only column/row TP: bit-preserving down columns, FP32 partial/host transport and sum then FP16; changed association separately teacher/state-qualified. Original FP16 draft and row-TP attention/GDN unchanged. Global HTTP cohort includes reset/admission/queue/reply/client archival; native timing excludes those. No microbenchmark-to-TPS inference.',model_qualified=True,source_sha256=actual,candidate_library_sha256=digest(candidate_library),path_promotions={},screen_ratios={},projection_profiles=dict(control='q8-residual',candidate='q8-residual-column-mlp'))
 paths=[('mtp2','scalar',2,'prefix',32,legacy),('ordinary256','batch',0,'replay',256,cohort_control),('mtp-batch','mtp-batch',1,'replay',32,cohort_control)]
 for name,mode,depth,recovery,rows,control_library in paths:
  selected_rows=rows;selected_profile='q8-residual';library=control_library
  control=experiment('screen-control-'+name,'venturi',mode,depth,'1,8',1,recovery)
  selected_profile='q8-residual-column-mlp';library=candidate_library
  candidate=experiment('screen-candidate-'+name,'venturi',mode,depth,'1,8',1,recovery)
  av={v['request_count']:v for v in control['summary']};bv={v['request_count']:v for v in candidate['summary']};metric='aggregate_complete_request_output_tps'
  ratios={str(n):bv[n][metric]/av[n][metric] for n in bv}
  report['screen_ratios'][name]=ratios;report['path_promotions'][name]=ratios['1']>1.01 or ratios['8']>1.02;save()
 # All screen arms finish before conditional matrices, so weak paths are visible.
 for name,mode,depth,recovery,rows,control_library in paths:
  if not report['path_promotions'][name]:continue
  selected_rows=rows;selected_profile='q8-residual';library=control_library
  experiment('repeat-control-'+name,'venturi',mode,depth,'1,2,4,8',3,recovery)
  selected_profile='q8-residual-column-mlp';library=candidate_library
  experiment('repeat-candidate-'+name,'venturi',mode,depth,'1,2,4,8',3,recovery)
 if any(report['path_promotions'].values()):
  experiment('repeat-llama-mtp2-fresh','llama','tensor',2,'1',3,'prefix')
  experiment('repeat-llama-nonspec-fresh','llama','tensor',0,'1,2,4,8',3)
 report['reused_controls']=[]
 for p in [root/'artifacts/venturi-batched-mtp-campaign-v3/campaign-report.json',root/'artifacts/venturi-batched-mtp-sdk-campaign-v8/campaign-report.json',root/'artifacts/venturi-wmma-campaign-v5/campaign-report.json',root/'artifacts/venturi-strided-external-depth-v9/campaign-report.json']:
  assert json.loads(p.read_text())['completed'];report['reused_controls'].append(dict(path=str(p),sha256=digest(p)))
 report['promotion_rule']='C1 HTTP >1% or C8 >2% admits full 3-repeat 1/2/4/8 path matrix; subsequent adoption also requires no repeated cell >2% regression and two improving qualifying trials. Full external completion is still >=5% on single post-first plus every matched cohort count.'
 report['external_goal_complete']=False;report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-native-column-mlp-campaign-v3',head+body)
