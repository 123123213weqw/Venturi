import subprocess
worker=r'''import pathlib,shutil,subprocess,os,traceback
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-llama-comparison-v1';src=root/'repos/codex-build/Venturi/llama-pair-r2-src-v1';original=root/'repos/codex-build/Venturi/llama-compare-src-v1'
try:
 assert not src.exists();shutil.copytree(original,src,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.ruff_cache'))
 p=src/'src/venturi/ops/csrc/blocks.cu';s=p.read_text();old='q8_tiled<T,2,1><<<n,256,0,stream>>>';assert s.count(old)==1;s=s.replace(old,'q8_tiled<T,2,2><<<1+(n-1)/2,256,0,stream>>>');p.write_text(s)
 env=os.environ.copy();env.update(PYTHONPATH=str(src/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with (out/'pair-build-library.txt').open('w') as f:subprocess.run(['/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python','-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(out/'pair-build')],cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
 rc=0
except BaseException:(out/'pair-build-failure.txt').write_text(traceback.format_exc());rc=1
(out/'pair-build-exit-code.txt').write_text(str(rc)+'\n')
'''
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-llama-comparison-v1');p=out/'pair-build-worker.py';assert not p.exists();p.write_text({worker!r})
with (out/'pair-build-worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,started_unix=time.time())))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
