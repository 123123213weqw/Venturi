import subprocess
worker=r'''import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/native-batch-src-v2';out=root/'artifacts/venturi-native-batch-v2';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_BATCH_TEST='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
def run(cmd,name,timeout=300):
 with (out/name).open('w') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
try:
 old=root/'artifacts/venturi-native-batch-v1';assert (old/'exit-code.txt').read_text().strip()=='1'
 paths=['blocks.cu','qwen_runtime.cuh','qwen_pipeline.cuh','qwen_mtp.cuh','qwen_batch.cuh'];checks={}
 for name in paths:
  path=pathlib.Path('src/venturi/ops/csrc')/name;new=(src/path).read_bytes();prior=(root/'repos/codex-build/Venturi/native-batch-src-v1'/path).read_bytes();assert new==prior;checks[str(path)]=hashlib.sha256(new).hexdigest()
 library=(old/'library.txt').read_text().strip();assert pathlib.Path(library).is_file();env['VENTURI_SM70_LIBRARY']=library
 (out/'library-reuse-proof.json').write_text(json.dumps(dict(native_source_sha256=checks,library=library,library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest()),indent=2));(out/'library.txt').write_text(library+'\n')
 run([py,'-m','pytest','-x','-q','tests/test_qwen_batch.py','--tb=short'],'batch-tests.log',timeout=180)
 run([py,'-m','pytest','-x','-q','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_pipeline.py','tests/test_qwen_mtp.py','--tb=short'],'scalar-regression-tests.log',timeout=240)
 run([py,'-m','pytest','-q','tests/test_completion_benchmark.py','--tb=short'],'admission-tests.log',timeout=90)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-batch-v2');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
