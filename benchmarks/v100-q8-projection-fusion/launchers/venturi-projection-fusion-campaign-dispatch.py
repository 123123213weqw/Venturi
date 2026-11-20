from pathlib import Path
import subprocess
base=Path('/tmp/venturi-tiled-staging-campaign-worker.py').read_text()
base=base.replace('venturi-tiled-staging-campaign-v4','venturi-projection-fusion-campaign-v4').replace('tiled-staging-src-v4','projection-fusion-src-v3').replace('venturi-tiled-staging-model-v3','venturi-projection-fusion-model-v3')
base=base.replace('tiled_staging=(library==candidate_library)', 'projection_group_mode=(2 if library==candidate_library else 1 if library==grouped_library else 0)')
start=base.index('candidate_library=library;')
end=base.index(" report['reused_external_controls']=[]",start)
base=base[:start]+'''candidate_library=library;baseline_library=root/'artifacts/venturi-tiled-staging-model-v3/build/398e4f4a1dd037ea/libventuri_blocks.so';grouped_library=root/'artifacts/venturi-projection-fusion-v1/build/b08a62aa2ac57cc2/libventuri_blocks.so';selected_profile='q8-residual'
try:
 assert digest(baseline_library)=='a6599e636cc0c018261310de7d83e1e8dd41d6edcc635f9161402a89f7775a3f'
 assert digest(grouped_library)=='6698f47bae3405279102890818782c992e8179aec966af8b5a08941160bb28b1'
 report['qualification_gates']=[]
 for name in ['venturi-projection-fusion-model-v2','venturi-projection-fusion-model-v3']:
  gate=root/'artifacts'/name;assert (gate/'exit-code.txt').read_text().strip()=='0';q=json.loads((gate/'qualification/report.json').read_text());assert q['completed'] and q['executed_candidate_profiles']==['q8-residual'] and q['profiles']['q8-residual']['same_profile_bytes_equal']
  mtp=json.loads((gate/'mtp-qualification/report.json').read_text());assert mtp['completed'] and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] for c in mtp['cases'])
  report['qualification_gates'].append(dict(path=str(gate),qualification_report_sha256=digest(gate/'qualification/report.json'),mtp_report_sha256=digest(gate/'mtp-qualification/report.json'),candidate_library_sha256=q['candidate_library_sha256'],unaffected_fp16_reference_reused=True,draft_cpu_reference_reused=True))
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Same-input projection grouping: mode0 pinned old fences, mode1 grouped transport with original kernels, mode2 grouped transport and fused complete row-dot kernels. Identical Q8-residual arithmetic. Fresh trained state gates precede matched screens and conditional three-repeat promotion. Prior strongest external controls explicitly reused, not selected downward.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),grouped_library_sha256=digest(grouped_library),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},tensor_core_model_dispatch=False);save()
 library=baseline_library;a=experiment('screen-control','venturi','batch',0,'1,2,8',1)
 library=grouped_library;b=experiment('screen-grouped','venturi','batch',0,'1,2,8',1)
 library=candidate_library;c=experiment('screen-fused','venturi','batch',0,'1,2,8',1)
 report['screen_gains']={v['label']:{str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],v['summary'])} for v in [b,c]};save()
 library=baseline_library;d=experiment('screen-mtp-control','venturi','scalar',1,'1',1)
 library=grouped_library;e=experiment('screen-mtp-grouped','venturi','scalar',1,'1',1)
 library=candidate_library;f=experiment('screen-mtp-fused','venturi','scalar',1,'1',1)
 report['mtp_screen_gains']={v['label']:v['summary'][0]['aggregate_complete_request_output_tps']/d['summary'][0]['aggregate_complete_request_output_tps'] for v in [e,f]}
 qualified=[]
 for tag,lib,v in [('grouped',grouped_library,b),('fused',candidate_library,c)]:
  g=report['screen_gains'][v['label']];mtp_gain=report['mtp_screen_gains']['screen-mtp-'+tag]
  if g['1']>1.015 and min(g['2'],g['8'],mtp_gain)>.99:qualified.append((g['1']*g['8']*mtp_gain,tag,lib))
 report['promotion_rule']='C1 >1.5%; no >1% C2/C8/MTP screen regression. Highest product of C1*C8*MTP HTTP ratios wins. Internal screening only; the >=5% external acceptance goal is unchanged.'
 if qualified:
  _,tag,chosen=max(qualified);report.update(promoted=True,promoted_variant=tag,promoted_library_sha256=digest(chosen));save()
  library=baseline_library;experiment('repeat-control','venturi','batch',0,'1,2,4,8',3)
  library=chosen;experiment('repeat-'+tag,'venturi','batch',0,'1,2,4,8',3)
  library=baseline_library;experiment('repeat-mtp-control','venturi','scalar',1,'1',3)
  library=chosen;experiment('repeat-mtp-'+tag,'venturi','scalar',1,'1',3)
 else:report['promoted']=False;save()
''' +base[end:]
Path('/tmp/venturi-projection-fusion-campaign-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-projection-fusion-campaign-v4');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
