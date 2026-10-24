from pathlib import Path
import subprocess
base=Path('/tmp/venturi-gdn-compact-campaign-worker.py').read_text().replace('venturi-gdn-compact-campaign-v3','venturi-prefill-budget-campaign-v2').replace('gdn-compact-src-v2','prefill-budget-src-v1').replace("prior=root/'artifacts/venturi-gdn-compact-model-v2'", "prior=root/'artifacts/venturi-prefill-budget-model-v1'")
base=base.replace('compact_gdn=(library==candidate_library)', 'projection_rows=selected_rows')
base=base.replace("cmd+=['--projection-profile',selected_profile]", "cmd+=['--projection-profile',selected_profile,'--projection-rows',str(selected_rows)]")
a=base.index('candidate_library=library;');b=base.index(" report['reused_external_controls']=[]",a)
base=base[:a]+'''candidate_library=library;baseline_library=pathlib.Path(library);selected_profile='q8-residual';selected_rows=32
try:
 assert digest(baseline_library)=='e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a'
 gate=root/'artifacts/venturi-prefill-budget-model-v1';report['qualification_gates']=[]
 for rows in [128,256]:
  path=gate/('qualification-'+str(rows))/'report.json';q=json.loads(path.read_text());assert q['completed'] and q['profiles']['q8-residual']['same_profile_bytes_equal'] and q['fixed_work']['max_tokens']==rows and q['fixed_work']['control_max_tokens']==32
  report['qualification_gates'].append(dict(projection_rows=rows,path=str(path),sha256=digest(path),default_fp16_reference_reused=True))
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Projection budget tuning only: same exact qualified native library/Q8 arithmetic/FP16 KV/context/work. Explicit rows32/128/256; fresh trained candidate byte/state gates before C1/C8 screens, conditional three-repeat whole matrix. MTP serving remains rows32 and prior results are explicitly reused, not remeasured.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},tensor_core_model_dispatch=False);save()
 selected_rows=32;a=experiment('screen-rows32','venturi','batch',0,'1,8',1)
 screens={}
 for rows in [128,256]:
  selected_rows=rows;b=experiment('screen-rows'+str(rows),'venturi','batch',0,'1,8',1)
  screens[str(rows)]={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])}
 report['screen_gains']=screens;eligible=[(g['8'],int(rows)) for rows,g in screens.items() if g['1']>.99 and g['8']>1.03]
 report['promotion_rule']='C1 no >1% regression, C8 >3% internal gain. Highest C8 ratio among eligible qualified budgets wins; then three-repeat full matrix. The >=5% external objective is unchanged.';report['promoted']=bool(eligible)
 if eligible:
  _,chosen=max(eligible);report['promoted_projection_rows']=chosen;save()
  selected_rows=32;experiment('repeat-rows32','venturi','batch',0,'1,2,4,8',3)
  selected_rows=chosen;experiment('repeat-rows'+str(chosen),'venturi','batch',0,'1,2,4,8',3)
 else:save()
 report['unchanged_mtp_reuse']=dict(path=str(root/'artifacts/venturi-gdn-compact-campaign-v3/campaign-report.json'),sha256=digest(root/'artifacts/venturi-gdn-compact-campaign-v3/campaign-report.json'),scope='Existing MTP1 rows32 code/library/configuration and measurements reused; no new MTP or 256-row MTP claim.')
''' +base[b:]
Path('/tmp/venturi-prefill-budget-campaign-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-prefill-budget-campaign-v2');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
