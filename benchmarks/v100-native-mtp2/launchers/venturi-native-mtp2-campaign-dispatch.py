from pathlib import Path
import subprocess
base=Path('/tmp/venturi-gdn-compact-campaign-worker.py').read_text().replace('venturi-gdn-compact-campaign-v3','venturi-native-mtp2-campaign-v3').replace('gdn-compact-src-v2','native-mtp2-src-v3').replace("prior=root/'artifacts/venturi-gdn-compact-model-v2'", "prior=root/'artifacts/venturi-native-mtp2-model-v2'")
base=base.replace('compact_gdn=(library==candidate_library)', 'new_mtp_library=(library==candidate_library)')
base=base.replace("if depth:cmd+=['--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer')]", "if depth:cmd+=['--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--mtp-depth',str(depth)]")
a=base.index('candidate_library=library;');b=base.index(" report['reused_external_controls']=[]",a)
base=base[:a]+'''candidate_library=library;baseline_library=root/'artifacts/venturi-gdn-compact-model-v2/build/0849ea2437e65c8a/libventuri_blocks.so';selected_profile='q8-residual'
try:
 assert digest(baseline_library)=='e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a'
 gate=root/'artifacts/venturi-native-mtp2-model-v2';q=json.loads((gate/'qualification/report.json').read_text());assert q['completed'] and q['draft_tokens']==2 and len(q['recurrence_reference'])==3 and all(x['passed'] for x in q['recurrence_reference'])
 assert len(q['cases'])==4 and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['committed_draft_state_equal_mtp1'] and c['pinned_target_trajectory_state_reused'] and c['mtp_stats']['forced_proposals']==0 for c in q['cases'])
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Native MTP2 replay-transaction implementation. Pinned prior MTP1 vs candidate MTP1 and MTP2, fresh C1/C8 screens after trained target/draft/recursive CPU gates. Ordinary batching, strongest external controls and prior quality gates explicitly reused. No forced proposals or precision changes.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),qualification_report_sha256=digest(gate/'qualification/report.json'),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},tensor_core_model_dispatch=False);save()
 library=baseline_library;a=experiment('screen-control-mtp1','venturi','scalar',1,'1,8',1)
 library=candidate_library;b=experiment('screen-candidate-mtp1','venturi','scalar',1,'1,8',1)
 library=candidate_library;c=experiment('screen-candidate-mtp2','venturi','scalar',2,'1,8',1)
 report['default_path_screen_ratios']={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])}
 gains={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],c['summary'])};report['screen_gains']=gains
 promote=gains['1']>1.01 or gains['8']>1.02;report['promoted']=promote;report['promotion_rule']='Quantify the full three-repeat matrix if MTP2 improves C1 >1% or C8 >2%; retain all regressions. This is not default enablement or external acceptance.';save()
 if promote:
  library=baseline_library;experiment('repeat-control-mtp1','venturi','scalar',1,'1,2,4,8',3)
  library=candidate_library;experiment('repeat-candidate-mtp2','venturi','scalar',2,'1,2,4,8',3)
 report['reused_best_native_batch']=dict(path=str(root/'artifacts/venturi-prefill-budget-campaign-v2/campaign-report.json'),sha256=digest(root/'artifacts/venturi-prefill-budget-campaign-v2/campaign-report.json'),scope='Qualified native ordinary rows256 batching remains the stronger parallel control; do not replace it with slower scalar MTP.')
''' +base[b:]
Path('/tmp/venturi-native-mtp2-campaign-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-mtp2-campaign-v3');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
