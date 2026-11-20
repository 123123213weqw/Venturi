import pathlib,subprocess
p=pathlib.Path('benchmarks/v100-native-residual/v100/campaign-v3/worker.py').read_text();prefix=p[:p.index("selected_profile='fp16'\ntry:")]
prefix=prefix.replace('venturi-model-profile-campaign-v3','venturi-shared-input-campaign-v3').replace('model-profile-src-v4','shared-input-src-v2').replace("prior=root/'artifacts/venturi-model-profile-fixtures-v1'","prior=root/'artifacts/venturi-shared-input-fixtures-v1'").replace('port=18947','port=18948')
prefix=prefix.replace("projection_profile=selected_profile if engine=='venturi' else None,completed=False)","projection_profile=selected_profile if engine=='venturi' else None,library_sha256=digest(library) if engine=='venturi' else None,shared_inputs=(library==candidate_library) if engine=='venturi' else None,completed=False)")
body=r'''
candidate_library=library;baseline_library=root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so';selected_profile='q8-residual'
try:
 gate=root/'artifacts/venturi-shared-input-qualification-v2';assert (gate/'exit-code.txt').read_text().strip()=='0';qualified=json.loads((gate/'qualification/report.json').read_text());assert qualified['completed'] and all(q['same_profile_bytes_equal'] for q in qualified['profiles'].values())
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Explicit same-input projection groups: unchanged arithmetic within each profile; retained cross-device output-reuse fences. Trained old/new byte gates before unprofiled 1/8 screening; conditional three-repeat matrix; separate diagnostics. Original/residual arithmetic are not interchangeable.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),qualification_report_sha256=digest(gate/'qualification/report.json'),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts});save()
 run([py,'tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',str(candidate_library),'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--skip-block-reference','--output-directory',str(out/'mtp-qualification')],out/'mtp-qualification.log',timeout=300)
 gains={}
 for profile in ['q8-residual','fp16']:
  selected_profile=profile;library=baseline_library;a=experiment('screen-unshared-'+profile,'venturi','batch',0,'1,8',1)
  library=candidate_library;b=experiment('screen-shared-'+profile,'venturi','batch',0,'1,8',1)
  gains[profile]={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])}
 report['screen_gains']=gains;save()
 selected_profile='q8-residual';library=baseline_library;experiment('screen-mtp-unshared','venturi','scalar',1,'1',1)
 library=candidate_library;experiment('screen-mtp-shared','venturi','scalar',1,'1',1)
 # Preset screening criteria; no external-speed success inferred from these.
 promote=min(gains['q8-residual'].values())>1.01;report['promoted']=promote;save()
 if promote:
  for profile in ['q8-residual','fp16']:
   selected_profile=profile;counts='1,2,4,8' if profile=='q8-residual' else '1,8'
   library=baseline_library;experiment('repeat-unshared-'+profile,'venturi','batch',0,counts,3)
   library=candidate_library;experiment('repeat-shared-'+profile,'venturi','batch',0,counts,3)
  selected_profile='q8-residual';library=baseline_library;experiment('repeat-mtp-unshared','venturi','scalar',1,'1',3)
  library=candidate_library;experiment('repeat-mtp-shared','venturi','scalar',1,'1',3)
  experiment('repeat-llama-tensor-off','llama','tensor',0,'1,2,4,8',3)
  try:experiment('repeat-llama-tensor-mtp1-c1','llama','tensor',1,'1',3)
  except Exception:report['external_mtp_failure_preserved']=True;save()
 report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
worker=prefix+body;compile(worker,'worker.py','exec')
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-shared-input-campaign-v3');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -'];r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
