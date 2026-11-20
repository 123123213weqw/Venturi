import subprocess,json
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server']
worker=r'''import os,pathlib,subprocess,json,hashlib,time,traceback
ROOT=pathlib.Path('/home/wzu/codex-build/Venturi');SRC=ROOT/'v100-occupancy-src-v1';OUT=ROOT/'v100-occupancy-prefill-v1'
PY='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
BASE=pathlib.Path((ROOT/'v100-pp-final-v1/library.txt').read_text().strip());old=json.loads((BASE.parent/'build.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(ROOT/'v100-tp-diagnostics-v2/python-deps')+':'+str(SRC/'src')+':'+str(SRC/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1',VENTURI_Q8_BASELINE_LIBRARY=str(BASE))
def run(cmd,log,timeout=900):
 with (OUT/log).open('w') as f:subprocess.run(cmd,cwd=SRC,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 assert json.loads((ROOT/'v100-occupancy-variants-v1/completed.json').read_text())['completed']
 csrc=SRC/'src/venturi/ops/csrc';d=OUT/'candidate';d.mkdir()
 assert all(digest(csrc/f)==h for f,h in old['source_sha256'].items())
 for f in ('blocks.cu','qwen_runtime.cuh','qwen_pipeline.cuh'):(d/f).write_bytes((csrc/f).read_bytes())
 f=d/'blocks.cu';s=f.read_text();needle='__global__ void q8_tiled(';assert s.count(needle)==1
 annotation='__global__\n#if defined(VENTURI_BLOCKS_ARCH) && VENTURI_BLOCKS_ARCH == 70\n__launch_bounds__(256, (Tokens > 1 ? 6 : 1))\n#endif\nvoid q8_tiled('
 f.write_text(s.replace(needle,annotation));library=d/'libventuri_blocks.so'
 command=list(old['command']);command[command.index(next(v for v in command if v.endswith('/blocks.cu')))]=str(f);command[-1]=str(library);command.insert(1,'--ptxas-options=-v')
 run(command,'candidate/build.log')
 manifest=dict(old,command=command,library_sha256=digest(library),source_sha256={p.name:digest(p) for p in d.iterdir() if p.suffix in ('.cu','.cuh')},experiment='SM70 only, q8_tiled launch bounds 6 for Tokens>1, 1 for decode; no math or geometry changes')
 (d/'build.json').write_text(json.dumps(manifest,indent=2));(OUT/'library.txt').write_text(str(library)+'\n')
 env['VENTURI_SM70_LIBRARY']=str(library)
 run([PY,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_pipeline.py','--tb=short'],'fixture-tests.log',timeout=300)
 run(['python3','tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',digest(library),'--output',str(OUT/'occupancy.json')],'occupancy.log')
 (OUT/'candidates.json').write_text(json.dumps([dict(name='prefill-only',library=str(library),token_tile=4,row_tile=4)],indent=2))
 run([PY,'tools/benchmark_q8_tiles.py',str(ROOT/'models/qwen3.5-27b-text-q8.ninfer'),'--baseline',str(BASE),'--candidates',str(OUT/'candidates.json'),'--output-directory',str(OUT/'projection-comparison'),'--gpu-index','1'],'projection-comparison.log')
 run([PY,'-u','tools/qualify_qwen_native.py',str(ROOT/'models/qwen3.5-27b-text-q8.ninfer'),'--expected-sha256','b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8','--blocks-library',str(library),'--previous-directory',str(ROOT/'greedy-validation-v3/full-model-run1'),'--previous-report-sha256','cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732','--tokenizer-file',str(ROOT/'models/qwen3.5-27b-bf16/tokenizer.json'),'--dtype','fp16','--weight-mode','resident','--pipeline-device','cuda:1','--pipeline-split-layer','32','--output-directory',str(OUT/'qualification')],'qualification.log',timeout=900)
 run([PY,'-m','pytest','-q','--tb=short'],'tests.log',timeout=600)
 for phase in range(4):
  selected=BASE if phase in (0,3) else library
  for mode in ('tp','pp'):
   run([PY,'-u','tools/benchmark_qwen_native_parallel.py',str(ROOT/'models/qwen3.5-27b-text-q8.ninfer'),'--library',str(selected),'--case',str(ROOT/'v100-pp-final-v1/case.json'),'--mode',mode,'--output-directory',str(OUT/f'phase-{phase}-{mode}'),'--concurrency','1,8','--repeats','1','--new-tokens','32','--split-layer','32'],f'phase-{phase}-{mode}.log')
 run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-final.txt')
 (OUT/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:
 (OUT/'failure.txt').write_text(traceback.format_exc());rc=1
(OUT/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
code=f'''import pathlib,subprocess,json,time
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-occupancy-prefill-v1');p.mkdir();(p/'worker.py').write_text({worker!r})
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
p=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,timeout=60);print(p.stdout,p.stderr);p.check_returncode()
