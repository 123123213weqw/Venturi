import os, pathlib, subprocess, json, hashlib, time, traceback
ROOT=pathlib.Path('/home/wzu/codex-build/Venturi')
SRC=ROOT/'v100-occupancy-src-v1'
OUT=ROOT/'v100-occupancy-variants-v1'
PY='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
BASE=pathlib.Path((ROOT/'v100-pp-final-v1/library.txt').read_text().strip())
base_manifest=json.loads((BASE.parent/'build.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(ROOT/'v100-tp-diagnostics-v2/python-deps')+':'+str(SRC/'src')+':'+str(SRC/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1',VENTURI_Q8_BASELINE_LIBRARY=str(BASE))
def run(cmd,log,timeout=900):
 with (OUT/log).open('w') as f:subprocess.run(cmd,cwd=SRC,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 (OUT/'started.json').write_text(json.dumps(dict(unix=time.time(),baseline_library=str(BASE),baseline_sha256=digest(BASE)),indent=2))
 assert digest(BASE)==base_manifest['library_sha256']
 csrc=SRC/'src/venturi/ops/csrc'
 assert all(digest(csrc/f)==h for f,h in base_manifest['source_sha256'].items())
 entries=[]
 for count in (6,8):
  d=OUT/f'minblocks-{count}';d.mkdir()
  for f in ('blocks.cu','qwen_runtime.cuh','qwen_pipeline.cuh'):(d/f).write_bytes((csrc/f).read_bytes())
  f=d/'blocks.cu';s=f.read_text();needle='__global__ void q8_tiled('
  assert s.count(needle)==1
  f.write_text(s.replace(needle,f'__global__ __launch_bounds__(256,{count}) void q8_tiled('))
  library=d/'libventuri_blocks.so'
  command=list(base_manifest['command'])
  command[command.index(next(v for v in command if v.endswith('/blocks.cu')))]=str(f)
  command[-1]=str(library)
  command.insert(1,'--ptxas-options=-v')
  run(command,f'minblocks-{count}/build.log')
  manifest=dict(base_manifest,command=command,library_sha256=digest(library),source_sha256={p.name:digest(p) for p in d.iterdir() if p.suffix in ('.cu','.cuh')},experiment=f'Only add __launch_bounds__(256,{count}) to q8_tiled; diagnostic candidate, not promoted')
  (d/'build.json').write_text(json.dumps(manifest,indent=2))
  env['VENTURI_SM70_LIBRARY']=str(library)
  run([PY,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_pipeline.py','--tb=short'],f'minblocks-{count}/fixture-tests.log',timeout=300)
  run(['python3','tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',digest(library),'--output',str(d/'occupancy.json')],f'minblocks-{count}/occupancy.log')
  entries.append(dict(name=f'minblocks-{count}',library=str(library),token_tile=4,row_tile=4))
 (OUT/'candidates.json').write_text(json.dumps(entries,indent=2))
 run([PY,'tools/benchmark_q8_tiles.py',str(ROOT/'models/qwen3.5-27b-text-q8.ninfer'),'--baseline',str(BASE),'--candidates',str(OUT/'candidates.json'),'--output-directory',str(OUT/'projection-comparison'),'--gpu-index','1'],'projection-comparison.log',timeout=900)
 run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-final.txt')
 (OUT/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:
 (OUT/'failure.txt').write_text(traceback.format_exc());rc=1
(OUT/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
