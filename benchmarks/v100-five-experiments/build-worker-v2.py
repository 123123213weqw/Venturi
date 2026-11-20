import hashlib,json,os,signal,subprocess,time
from pathlib import Path
S=Path('/home/data/wangyue/repos/codex-build/Venturi/five-src-v4')
O=Path('/home/data/wangyue/artifacts/venturi-five-v2');O.mkdir(exist_ok=False)
P='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
E=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{S}/src:{S}/tests:/home/wzu/.local/lib/python3.10/site-packages')
def save(n,j):(O/n).write_text(json.dumps(j,indent=2)+'\n')
def run(n,args,limit):
 with (O/(n+'.log')).open('xb') as f:
  p=subprocess.Popen(args,cwd=S,env=E,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=limit)
  except BaseException:
   os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5);raise
 save(n+'.exit.json',dict(rc=rc,args=args));assert rc==0,n
save('source-pins.json',{str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (S/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
run('build',[P,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--q8-row-pair','--q8-swiglu-quant','--q8-rms-quant','--nvcc','/usr/local/cuda/bin/nvcc','--output-dir',str(O/'build')],240)
lib=(O/'build.log').read_text().strip().splitlines()[-1];save('library.json',dict(library=lib))
E['VENTURI_EXPERIMENT_LIBRARY']=lib
run('fixtures',[P,'-m','pytest','-xq','tests/test_native_experiments.py'],180)
save('completed.json',dict(completed=True,library=lib,unix=time.time()))
