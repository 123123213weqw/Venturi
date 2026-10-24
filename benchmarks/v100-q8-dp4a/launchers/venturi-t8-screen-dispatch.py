import pathlib,subprocess
old=pathlib.Path('benchmarks/v100-native-batch/launchers/venturi-batch-campaign-worker-v4.py').read_text();prefix=old[:old.index('try:\n assert (prior')]
prefix=prefix.replace("out=root/'artifacts/venturi-native-batch-v4'","out=root/'artifacts/venturi-native-t8-screen-v1'").replace("src=root/'repos/codex-build/Venturi/native-batch-src-v3'","src=root/'repos/codex-build/Venturi/native-t8-src-v1'")
body=r'''
try:
 baseline=library;report['scope']='Exact-order T8 ablation: original T4/R4 library, naive old T8 six-block bound, and adaptive T8 four-block bound. Source and precision pinned, no default promotion from microbenchmarks.'
 report['baseline_library_sha256']=digest(baseline);report['source_hashes']={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};save()
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-before.txt')
 for name,source in [('naive',root/'repos/codex-build/Venturi/native-batch-src-v3'),('adaptive',src)]:
  env['PYTHONPATH']=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(source/'src')+':'+str(src/'tests')
  run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--q8-token-tile','8','--output-dir',str(out/'build')],out/f'build-{name}.log',timeout=180)
  path=pathlib.Path((out/f'build-{name}.log').read_text().splitlines()[-1]);assert path.is_file();report[name+'_library']=str(path);report[name+'_sha256']=digest(path);save()
 env['PYTHONPATH']=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests');library=pathlib.Path(report['adaptive_library']);env.update(VENTURI_SM70_LIBRARY=str(library),VENTURI_Q8_BASELINE_LIBRARY=str(baseline),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_batch.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_mtp.py','tests/test_qwen_pipeline.py','--tb=short'],out/'fixture-tests.log',timeout=180)
 candidates=[dict(name='naive-t8-b6',token_tile=8,row_tile=4,library=report['naive_library']),dict(name='adaptive-t8-b4',token_tile=8,row_tile=4,library=report['adaptive_library'])];(out/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
 run([py,'-u','tools/benchmark_q8_tiles.py',str(target),'--baseline',str(baseline),'--candidates',str(out/'candidates.json'),'--token-counts','8,32','--gpu-index','1','--output-directory',str(out/'projection')],out/'projection.log',timeout=300)
 for name in ['naive','adaptive']:
  path=pathlib.Path(report[name+'_library']);run([py,'tools/inspect_cuda_occupancy.py','--library',str(path),'--expected-sha256',digest(path),'--output',str(out/f'{name}-resources.json')],out/f'{name}-resources.log',timeout=90)
 # Preserve end-to-end gating; a projection win alone is not a promotion.
 library=baseline;a=experiment('baseline-batch','venturi','batch',0,'1,8',1)
 library=pathlib.Path(report['adaptive_library']);b=experiment('adaptive-batch','venturi','batch',0,'1,8',1)
 report['screen_gains']={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])}
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt');report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
worker=prefix+body
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-t8-screen-v1');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
