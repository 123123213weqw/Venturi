import pathlib,subprocess
old=pathlib.Path('benchmarks/v100-native-batch/launchers/venturi-batch-campaign-worker-v4.py').read_text();prefix=old[:old.index('try:\n assert (prior')]
prefix=prefix.replace("out=root/'artifacts/venturi-native-batch-v4'","out=root/'artifacts/venturi-tp-worker-screen-v1'").replace("src=root/'repos/codex-build/Venturi/native-batch-src-v3'","src=root/'repos/codex-build/Venturi/tp-worker-src-v1'")
body=r'''
try:
 baseline=library;report['scope']='Experimental persistent peer CPU submitter. Exact Q8/FP16 arithmetic unchanged; serial remains default. Matched unprofiled 1/8 HTTP screen, no kernel or precision change in the engine.'
 assert (root/'artifacts/venturi-dp4a-projection-v2/exit-code.txt').read_text().strip()=='0'
 report['baseline_library_sha256']=digest(baseline);report['source_hashes']={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};save()
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--tp-dispatch','worker','--output-dir',str(out/'build')],out/'build.log',timeout=180)
 library=pathlib.Path((out/'build.log').read_text().splitlines()[-1]);assert library.is_file();(out/'library.txt').write_text(str(library)+'\n');report['candidate_library_sha256']=digest(library);save()
 env.update(VENTURI_SM70_LIBRARY=str(library),VENTURI_Q8_BASELINE_LIBRARY=str(baseline),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_batch.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_mtp.py','tests/test_qwen_pipeline.py','--tb=short'],out/'fixture-tests.log',timeout=180)
 candidate=library;library=baseline;a=experiment('baseline-batch','venturi','batch',0,'1,8',1)
 library=candidate;b=experiment('worker-batch','venturi','batch',0,'1,8',1)
 report['screen_gains']={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])}
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt');report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
worker=prefix+body
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-tp-worker-screen-v1');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
