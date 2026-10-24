import pathlib,subprocess
s=pathlib.Path('benchmarks/v100-shared-inputs/v100/campaign-v3/worker.py').read_text();prefix=s[:s.index('candidate_library=library;')]
prefix=prefix.replace('venturi-shared-input-campaign-v3','venturi-wmma-campaign-v5').replace('shared-input-src-v2','wmma-src-v5').replace("prior=root/'artifacts/venturi-shared-input-fixtures-v1'","prior=root/'artifacts/venturi-wmma-guarded-v4'").replace('port=18948','port=18949').replace('shared_inputs=(library==candidate_library)', 'guarded_t8=(library==candidate_library)')
body=r'''
candidate_library=library;baseline_library=root/'artifacts/venturi-shared-input-fixtures-v1/build/77fedac214853798/libventuri_blocks.so';selected_profile='q8-residual'
try:
 gate=root/'artifacts/venturi-wmma-guarded-v4';assert (gate/'exit-code.txt').read_text().strip()=='0';qualified=json.loads((gate/'qualification/report.json').read_text());assert qualified['completed'] and all(q['same_profile_bytes_equal'] for q in qualified['profiles'].values())
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Guarded V100 DP4A T8 scheduling, same original/residual math and shared-input ownership. WMMA standalone prototype remains NOT selected by native model. Exact trained old/new bytes precede 1/8 screening; conditional repeat matrix; no external win presumption.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),qualification_report_sha256=digest(gate/'qualification/report.json'),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},tensor_core_model_dispatch=False);save()
 prior_mtp=root/'artifacts/venturi-shared-input-campaign-v3/mtp-qualification/report.json';assert json.loads(prior_mtp.read_text())['completed'];report['reused_trained_mtp_state_reference']=dict(path=str(prior_mtp),sha256=digest(prior_mtp),scope='Unchanged MTP/draft and unchanged T1/T2 target dispatch; fresh trained byte gate includes changed T8 prefill. Fresh native MTP timings below still enforce pinned outputs and report acceptance. New final suite retests forced transaction boundaries. Not a new independent draft-reference run.');save()
 library=baseline_library;a=experiment('screen-control','venturi','batch',0,'1,8',1)
 library=candidate_library;b=experiment('screen-t8','venturi','batch',0,'1,8',1)
 gains={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])};report['screen_gains']=gains;save()
 promote=gains['1']>1.005 and gains['8']>1.05;report['promoted']=promote;save()
 if promote:
  library=baseline_library;experiment('repeat-control','venturi','batch',0,'1,2,4,8',3)
  library=candidate_library;experiment('repeat-t8','venturi','batch',0,'1,2,4,8',3)
  library=baseline_library;experiment('repeat-mtp-control','venturi','scalar',1,'1',3)
  library=candidate_library;experiment('repeat-mtp-t8','venturi','scalar',1,'1',3)
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
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-wmma-campaign-v5');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -'];r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
