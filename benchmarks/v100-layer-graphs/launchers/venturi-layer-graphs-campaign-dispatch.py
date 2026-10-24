from pathlib import Path
import runpy
s=Path('/tmp/venturi-token-order-campaign-v6-worker.py').read_text().replace('token-order-src-v2','layer-graphs-src-v3').replace('venturi-token-order-model-v5','venturi-layer-graphs-model-v4').replace('venturi-token-order-campaign-v6','venturi-layer-graphs-campaign-v5')
s=s.replace("    cmd+=['--projection-profile',selected_profile,'--projection-rows',str(selected_rows)]", "    cmd+=['--projection-profile',selected_profile,'--projection-rows',str(selected_rows)]")
a="   if depth:cmd+=['--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--mtp-depth',str(depth),'--mtp-recovery',recovery]";assert a in s;s=s.replace(a,a+"\n   if library==candidate_library and enable_graph:cmd+=['--target-layer-graph']",1)
a=s.index('  q=json.loads((prior/\'mtp-batch-qualification/report.json\')');b=s.index('  build=json.loads',a)
s=s[:a]+'''  q=json.loads((prior/'graph-qualification/report.json').read_text());assert q['completed'] and len(q['cases'])==4 and q['reset_repeat']['executable_reused'] and all(c['all_verification_fp16_bytes_equal'] and c['committed_target_state_equal'] and c['committed_draft_state_equal'] for c in q['cases'])
  q=json.loads((prior/'mtp2-qualification/report.json').read_text());assert q['completed'] and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['committed_draft_state_equal_mtp1'] for c in q['cases'])
''' +s[b:]
a=s.index('  paths=[');b=s.index(" report['reused_controls']=[]",a)
s=s[:a]+'''  selected_rows=32;enable_graph=False;library=legacy
  control=experiment('screen-control-mtp2','venturi','scalar',2,'1,8',1,'prefix')
  library=candidate_library
  experiment('screen-same-library-eager','venturi','scalar',2,'1,8',1,'prefix')
  enable_graph=True
  candidate=experiment('screen-candidate-mtp2','venturi','scalar',2,'1,8',1,'prefix')
  ix=lambda e:{v['request_count']:v for v in e['summary']};av,bv=ix(control),ix(candidate);metric='aggregate_complete_request_output_tps';gains={str(n):bv[n][metric]/av[n][metric] for n in bv};promote=gains['1']>1.01 or gains['8']>1.02
  report['screen_ratios']['mtp2']=gains;report['path_promotions']['mtp2']=promote;save()
  if promote:
   library=legacy;enable_graph=False;experiment('repeat-control-mtp2','venturi','scalar',2,'1,2,4,8',3,'prefix')
   library=candidate_library;enable_graph=True;experiment('repeat-candidate-mtp2','venturi','scalar',2,'1,2,4,8',3,'prefix')
   # Recheck the pinned competitor, retaining its best supported single MTP2
   # setting and the stable non-speculative parallel matrix; no artificial cap.
   experiment('repeat-llama-mtp2-fresh','llama','tensor',2,'1',3,'prefix')
   experiment('repeat-llama-nonspec-fresh','llama','tensor',0,'1,2,4,8',3)
''' +s[b:]
s=s.replace("selected_profile='q8-residual';selected_rows=32", "selected_profile='q8-residual';selected_rows=32;enable_graph=False")
s=s.replace('Adjacent token-tile CTA ordering only, original arithmetic/KV/draft/host-TP protocol and optimized SDK unchanged.', 'Explicit request-owned scalar MTP2 target-layer graphs with device-positioned KV/RoPE/attention and original arithmetic/KV/draft/host-TP fences. Other workloads remain eager; no parallel graph claim.')
s=s.replace('C1 HTTP >1% or C8 >2%; MTP1 cohort C2 >2% also admits its full matrix.', 'C1 HTTP >1% or C8 >2% admits the scalar MTP2 full matrix; separate same-library eager screen and fresh pinned llama controls.')
runpy.run_path('/tmp/venturi-dispatch-helper.py')['dispatch']('venturi-layer-graphs-campaign-v5',s)
