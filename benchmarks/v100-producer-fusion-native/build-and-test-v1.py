"""Build and fixture-qualify OFF/ON native libraries without stopping production."""
import hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
source=Path('/home/data/wangyue/repos/codex-build/Venturi/producer-fusion-native-src-v4')
out=Path('/home/data/wangyue/artifacts/venturi-producer-fusion-native-v1')
python='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
control='/home/data/wangyue/artifacts/venturi-q8-prefetch-v3/build/f634328d5b15400b/libventuri_blocks.so'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
 PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{source}/src:{source}/tests:/home/wzu/.local/lib/python3.10/site-packages')
def save(name,j):(out/name).write_text(json.dumps(j,indent=2)+'\n')
def audit(name):
 a=json.loads(subprocess.check_output(['docker','inspect','qwen38-27b'],timeout=15))[0]
 j=dict(container_id=a['Id'],started_at=a['State']['StartedAt'],restart_count=a['RestartCount'],status=a['State']['Status'],health=a['State']['Health']['Status'],
  canonical_config_fingerprint=hashlib.sha256(json.dumps(dict(args=a['Args'],env=a['Config']['Env'],mounts=sorted(a['Mounts'],key=lambda v:(v.get('Destination',''),v.get('Source',''),v.get('Type','')))),sort_keys=True).encode()).hexdigest(),
  counter_policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s],
  health_api=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5)),unix=time.time())
 save(name,j);assert j['status']=='running' and j['health']=='healthy';return j

def run(name,args,limit):
 save(name+'.command.json',args)
 with (out/(name+'.log')).open('xb') as f:
  p=subprocess.Popen(args,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 (out/(name+'.exit')).write_text(str(rc)+'\n');assert rc==0,name

def idle():
 for _ in range(30):
  slots=json.load(urllib.request.urlopen('http://127.0.0.1:8000/slots',timeout=5))
  if not any(s.get('is_processing') for s in slots):return
  time.sleep(2)
 raise TimeoutError('Production busy; do not interrupt')

out.mkdir(parents=True,exist_ok=False)
try:
 before=audit('production-before.json');idle()
 pins={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (source/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};save('executed-source-sha256.json',pins)
 run('build-api-tests',[python,'-m','pytest','-q','tests/test_ada_build.py'],60)
 libraries=[]
 for enabled in [False,True]:
  label='on' if enabled else 'off'
  run('build-'+label,[python,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--q8-row-pair','--nvcc','/usr/local/cuda/bin/nvcc','--output-dir',str(out/'build')]+(['--q8-swiglu-quant'] if enabled else []),240)
  lib=Path((out/('build-'+label+'.log')).read_text().strip().splitlines()[-1]);assert lib.is_file();libraries.append(str(lib))
 save('libraries.json',dict(off=libraries[0],on=libraries[1],pinned_control=control))
 env.update(VENTURI_SWIGLU_BASELINE=libraries[0],VENTURI_SWIGLU_NATIVE=libraries[1])
 idle();run('native-fusion-tests',[python,'-m','pytest','-q','tests/test_native_swiglu_quant.py'],240)
 env.update(VENTURI_SQ_LIBRARY=libraries[1],VENTURI_SQ_CONTROL=control)
 idle();run('producer-byte-tests',[python,'-m','pytest','-q','tests/test_swiglu_quant.py'],90)
 env.update(VENTURI_NATIVE_COLUMN_MLP_LIBRARY=libraries[1],VENTURI_NATIVE_HEAD_GDN_LIBRARY=libraries[1])
 idle();run('native-fallback-tests',[python,'-m','pytest','-q','tests/test_native_column_mlp.py','tests/test_native_head_gdn.py'],240)
 save('completed.json',dict(completed=True,production_deployed=False,libraries=libraries,ended_unix=time.time()))
finally:
 after=audit('production-after.json')
 if 'before' in globals():
  assert all(before[k]==after[k] for k in ['container_id','started_at','restart_count','canonical_config_fingerprint','counter_policy'])
