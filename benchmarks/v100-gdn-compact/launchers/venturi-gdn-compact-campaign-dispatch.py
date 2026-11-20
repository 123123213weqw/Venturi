from pathlib import Path
import subprocess
base=Path('/tmp/venturi-tiled-staging-campaign-worker.py').read_text().replace('venturi-tiled-staging-campaign-v4','venturi-gdn-compact-campaign-v3').replace('tiled-staging-src-v4','gdn-compact-src-v2').replace('venturi-tiled-staging-model-v3','venturi-gdn-compact-model-v2')
base=base.replace('tiled_staging=(library==candidate_library)','compact_gdn=(library==candidate_library)')
a=base.index('candidate_library=library;');b=base.index(" report['reused_external_controls']=[]",a)
base=base[:a]+'''candidate_library=library;baseline_library=root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so';selected_profile='q8-residual'
try:
 assert digest(baseline_library)=='698452e2b2a39f3a5bb4d08d58e37a3330857f6444cc462f96147f53a7ab5e3d'
 gate=root/'artifacts/venturi-gdn-compact-model-v2';q=json.loads((gate/'qualification/report.json').read_text());assert q['completed'] and q['executed_candidate_profiles']==['q8-residual'] and q['profiles']['q8-residual']['same_profile_bytes_equal']
 mtp=json.loads((gate/'mtp-qualification/report.json').read_text());assert mtp['completed'] and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] for c in mtp['cases'])
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Residual target compact GDN geometry preserves original eight logical warp partials/output/state bytes. Default FP16/SM89/DK256 unchanged. Fresh candidate trained/MTP gates precede C1/C8 screening and conditional three-repeat matrix/MTP. Strongest pinned external controls explicitly reused.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),qualification_report_sha256=digest(gate/'qualification/report.json'),mtp_report_sha256=digest(gate/'mtp-qualification/report.json'),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},tensor_core_model_dispatch=False);save()
 library=baseline_library;a=experiment('screen-control','venturi','batch',0,'1,8',1)
 library=candidate_library;b=experiment('screen-compact','venturi','batch',0,'1,8',1)
 gains={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])};report['screen_gains']=gains
 promote=gains['1']>1.01 and gains['8']>1.03;report['promotion_rule']='C1 >1% and C8 >3% internal screen improvement. Conditional whole matrix/MTP uses three repeats. This does not replace >=5% external acceptance.';report['promoted']=promote;save()
 if promote:
  library=baseline_library;experiment('repeat-control','venturi','batch',0,'1,2,4,8',3)
  library=candidate_library;experiment('repeat-compact','venturi','batch',0,'1,2,4,8',3)
  library=baseline_library;experiment('repeat-mtp-control','venturi','scalar',1,'1',3)
  library=candidate_library;experiment('repeat-mtp-compact','venturi','scalar',1,'1',3)
''' +base[b:]
Path('/tmp/venturi-gdn-compact-campaign-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-gdn-compact-campaign-v3');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
