import subprocess
code=r'''import pathlib,subprocess,time,json,os,hashlib,traceback
root=pathlib.Path('/home/wzu/codex-build/Venturi');out=root/'v100-occupancy-final-check-v1';out.mkdir()
source=root/'v100-occupancy-promoted-src-v1'
try:
 terminal=root/'v100-occupancy-diagnostics-v1/exit-code.txt'
 for _ in range(120):
  if terminal.exists():break
  time.sleep(10)
 else:raise RuntimeError('Bounded wait elapsed, no postcheck GPU jobs started')
 assert terminal.read_text().strip()=='0'
 library=pathlib.Path((root/'v100-occupancy-prefill-v2/library.txt').read_text().strip())
 manifest=json.loads((library.parent/'build.json').read_text())
 files={f:hashlib.sha256((source/'src/venturi/ops/csrc'/f).read_bytes()).hexdigest() for f in manifest['source_sha256']}
 assert files==manifest['source_sha256']
 assert hashlib.sha256(library.read_bytes()).hexdigest()==manifest['library_sha256']
 (out/'source-and-library.json').write_text(json.dumps(dict(source=str(source),source_sha256=files,library_sha256=manifest['library_sha256'],exact_executed_native_source_match=True),indent=2))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'v100-tp-diagnostics-v2/python-deps')+':'+str(source/'src')+':'+str(source/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1',VENTURI_SM70_LIBRARY=str(library),VENTURI_Q8_BASELINE_LIBRARY=(root/'v100-pp-final-v1/library.txt').read_text().strip())
 with (out/'tests.log').open('w') as log:subprocess.run(['/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python','-m','pytest','-x','-q','tests/test_cuda_occupancy_tool.py','tests/test_q8_tiling.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_pipeline.py','--tb=short'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 with (out/'gpu-final.txt').open('w') as log:subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],stdout=log,check=True)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:
 (out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
launcher=f'''import pathlib,subprocess,json,time
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-occupancy-postcheck-dispatch');p.mkdir();(p/'worker.py').write_text({code!r})
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=launcher,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
