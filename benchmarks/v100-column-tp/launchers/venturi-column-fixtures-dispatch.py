from pathlib import Path
import subprocess,json,hashlib,sys
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-column-tp';snap='column-tp-src-v1';remote='/home/data/wangyue/repos/codex-build/Venturi/'+snap
cmd=ns['SSH'][:-1]+['mkdir -p '+remote];subprocess.run(cmd,check=True,timeout=60)
ssh=' '.join(ns['SSH'][:-2]);flags=['--exclude='+v for v in ['.git/','target/','node_modules/','.env','.env.*','.venv*','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']]
r=subprocess.run(['rsync','-az',*flags,'-e',ssh,'src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/'],capture_output=True,text=True,timeout=180);(out/'sync-v1.log').write_text(r.stdout+r.stderr);r.check_returncode()
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};(out/'source-manifest-v1.json').write_text(json.dumps(expected,indent=2)+'\n')
code='''import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/column-tp-src-v1';out=root/'artifacts/venturi-column-tp-fixtures-v1';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def run(cmd,name,timeout=600):
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\\n')
 with (out/name).open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
try:
 expected=EXPECTED
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert actual==expected,'Full source manifest mismatch before build'
 (out/'pre-build-source.json').write_text(json.dumps(actual,indent=2)+'\\n')
 run([py,'-c',"from venturi.ops.build import build_blocks_backend;from pathlib import Path;p=build_blocks_backend('"+str(out/'build')+"',nvcc='/usr/local/cuda/bin/nvcc',architecture='sm70');Path('"+str(out/'library.txt')+"').write_text(str(p)+'\\\\n')"],'build.log',timeout=600)
 library=(out/'library.txt').read_text().strip();env['VENTURI_COLUMN_TP_LIBRARY']=library
 run([py,'-m','pytest','-x','-q','tests/test_q8_column_tp.py','tests/test_ada_build.py','--tb=short'],'focused-tests.log',timeout=600)
 BASEENVS
 env.pop('VENTURI_TILE_ORDER_LIBRARY',None);env.pop('VENTURI_T16_LIBRARY',None);env.pop('VENTURI_LAYER_GRAPH_LIBRARY',None)
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=600)
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),'--output',str(out/'resources.json')],'resources.log',timeout=180)
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+'\\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\\n');raise SystemExit(rc)
'''
# Reuse unchanged fixture environment from the preceding exact library campaign.
old=Path('/tmp/venturi-layer-graphs-fixtures-v3-worker.py').read_text();start=old.index(' env.update(VENTURI_DP4A_LIBRARY=library');end=old.index(" run([py,'-m','pytest','-x'",start);block=old[start:end];block='\n'.join(l for l in block.splitlines() if "env['VENTURI_LAYER_GRAPH_LIBRARY']" not in l)
block=" baseline=root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so'\n"+block
code=code.replace(' expected=EXPECTED',' expected='+repr(expected)).replace(' BASEENVS',block)
# Check the path-writing Python command before dispatch, not by running a compiler locally.
compile(code,'worker','exec');ns['dispatch']('venturi-column-tp-fixtures-v1',code)
