import subprocess
worker=r'''import pathlib,subprocess,os,json,traceback,hashlib,signal,time,statistics
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-llama-comparison-v1';src=root/'repos/codex-build/Venturi/llama-pair-r2-src-v1';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');d=out/'pair-screen-v1';d.mkdir()
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run(cmd,name,timeout=600):
 with (d/name).open('w') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:stop(p);raise
  if rc:raise RuntimeError(f'Owned candidate command exited {rc}: {name}')
try:
 assert (out/'recovery-v2/screen-exit-code.txt').exists(),'External screen still running; no GPU experiment started'
 assert (out/'pair-build-exit-code.txt').read_text().strip()=='0'
 r=json.loads((out/'recovery-v2/screen-report.json').read_text());assert any(x['completed'] and x['engine']=='llama' for x in r['experiments'])
 library=pathlib.Path((out/'pair-build-library.txt').read_text().splitlines()[-1]);baseline=root/'repos/codex-build/Venturi/v100-mtp-build/fb11e6eff12b2fbf/libventuri_blocks.so'
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_SM70_LIBRARY=str(library),VENTURI_Q8_BASELINE_LIBRARY=str(baseline),VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_mtp.py','--tb=short'],'tests.log',timeout=180)
 candidates=[dict(name='verification-rows-2',token_tile=4,row_tile=4,library=str(library))];(d/'candidates.json').write_text(json.dumps(candidates,indent=2))
 run([py,'tools/benchmark_q8_tiles.py',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'),'--baseline',str(baseline),'--candidates',str(d/'candidates.json'),'--verification-pairs','--gpu-index','1','--output-directory',str(d/'projection')],'projection.log')
 run(['python3','tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',hashlib.sha256(library.read_bytes()).hexdigest(),'--output',str(d/'occupancy.json')],'occupancy.log')
 micro=json.loads((d/'projection/report.json').read_text());assert micro['completed'] and all(c['exact_baseline_output_bytes'] for c in micro['cases'])
 rows=[dict(shape=c['shape'],ratio=c['median_gpu_event_ms']['baseline']/c['median_gpu_event_ms']['verification-rows-2']) for c in micro['cases'] if c['tokens']==2]
 summary=dict(completed=True,variant='SM70 T2 uses two output rows per block; unchanged per-output FP32 FMA/reduction ordering',ratios=rows,median_speed_ratio=statistics.median(r['ratio'] for r in rows),default_changed=False,scope='Projection-only screen, not engine TPS or an external win',native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
 (d/'report.json').write_text(json.dumps(summary,indent=2)+'\n');rc=0
except BaseException:(d/'failure.txt').write_text(traceback.format_exc());rc=1
(d/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-llama-comparison-v1');p=out/'pair-screen-worker.py';assert not p.exists();p.write_text({worker!r})
with (out/'pair-screen-worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'pair-screen-process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
