import pathlib,subprocess
old=pathlib.Path('benchmarks/v100-native-batch/launchers/venturi-batch-campaign-worker-v4.py').read_text();prefix=old[:old.index('try:\n assert (prior')]
prefix=prefix.replace("out=root/'artifacts/venturi-native-batch-v4'","out=root/'artifacts/venturi-model-profile-campaign-v3'").replace("src=root/'repos/codex-build/Venturi/native-batch-src-v3'","src=root/'repos/codex-build/Venturi/model-profile-src-v4'").replace("prior=root/'artifacts/venturi-native-batch-v2'","prior=root/'artifacts/venturi-model-profile-fixtures-v1'").replace('port=18944','port=18947')
prefix=prefix.replace("item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,completed=False)","item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,projection_profile=selected_profile if engine=='venturi' else None,completed=False)")
prefix=prefix.replace("   if mode=='batch':cmd+=['--batch-target','--batch-window-ms','25']", "   cmd+=['--projection-profile',selected_profile]\n   if depth:cmd+=['--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer')]\n   if mode=='batch':cmd+=['--batch-target','--batch-window-ms','25']")
body=r'''
selected_profile='fp16'
try:
 gate=root/'artifacts/venturi-model-profile-qualification-v2';assert (gate/'exit-code.txt').read_text().strip()=='0';qualified=json.loads((gate/'qualification/report.json').read_text());assert qualified['completed']
 build=json.loads(library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Matched native default versus explicit residual-activation arithmetic. Trained CPU/trajectory and same-profile cache gates precede timing. Same ABI16 library, original weights, FP16 KV, formatted 23 IDs and 32 fixed outputs. Screening then conditional three-repeat matrix, no external-win presumption.',library_sha256=digest(library),qualification_report_sha256=digest(gate/'qualification/report.json'),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts});save()
 run([py,'tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',str(library),'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--skip-block-reference','--output-directory',str(out/'mtp-qualification')],out/'mtp-qualification.log',timeout=300)
 report['unchanged_draft_reference_reused_from']='benchmarks/v100-native-mtp/v100/final-v4/qualification-tp/report.json; separate original-source FP64 draft block gate; draft stays original FP16 arithmetic';save()
 selected_profile='fp16';baseline=experiment('screen-original','venturi','batch',0,'1,8',1)
 selected_profile='q8-residual';candidate=experiment('screen-residual','venturi','batch',0,'1,8',1)
 gains={str(a['request_count']):b['aggregate_complete_request_output_tps']/a['aggregate_complete_request_output_tps'] for a,b in zip(baseline['summary'],candidate['summary'])};report['screen_gains']=gains;save()
 selected_profile='fp16';experiment('screen-mtp-original','venturi','scalar',1,'1',1)
 selected_profile='q8-residual';experiment('screen-mtp-residual','venturi','scalar',1,'1',1)
 if gains['1']>1.02 and gains['8']>1.1:
  selected_profile='fp16';experiment('repeat-original','venturi','batch',0,'1,2,4,8',3)
  selected_profile='q8-residual';experiment('repeat-residual','venturi','batch',0,'1,2,4,8',3)
  selected_profile='fp16';experiment('repeat-mtp-original','venturi','scalar',1,'1',3)
  selected_profile='q8-residual';experiment('repeat-mtp-residual','venturi','scalar',1,'1',3)
  experiment('repeat-llama-tensor-off','llama','tensor',0,'1,2,4,8',3)
  try:experiment('repeat-llama-tensor-mtp1-c1','llama','tensor',1,'1',3)
  except Exception:report['external_mtp_c1_failure_preserved']=True;save()
 report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
worker=prefix+body;compile(worker,'worker.py','exec')
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-model-profile-campaign-v3');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -'];r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
