import pathlib,subprocess,json,os,time,hashlib,traceback
root=pathlib.Path('/home/wzu/codex-build/Venturi');src=root/'v100-mtp-src-v4';out=root/'v100-mtp-final-v4'
py='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
def run(cmd,name,timeout=900):
 with (out/name).open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 terminal=root/'v100-mtp-qualification-v3/exit-code.txt'
 for _ in range(120):
  if terminal.exists():break
  time.sleep(10)
 else:raise RuntimeError('Bounded wait elapsed; final experiment not started')
 assert terminal.read_text().strip()=='0','Initial trained qualification failed'
 baseline=(root/'v100-mtp-qualification-v3/library.txt').read_text().strip()
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(root/'v100-mtp-build')],'library.txt')
 library=(out/'library.txt').read_text().strip();env['VENTURI_SM70_LIBRARY']=library;env['VENTURI_Q8_BASELINE_LIBRARY']=baseline
 run([py,'-m','pytest','-x','-q','tests/test_q8_tiling.py','tests/test_qwen_mtp.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_pipeline.py','--tb=short'],'fixture-tests.log',timeout=300)
 (out/'candidates.json').write_text(json.dumps([dict(name='two-token-verification',token_tile=4,row_tile=4,library=library)],indent=2))
 run([py,'tools/benchmark_q8_tiles.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--baseline',baseline,'--candidates',str(out/'candidates.json'),'--output-directory',str(out/'projection-comparison'),'--gpu-index','1','--verification-pairs'],'projection-comparison.log')
 run(['python3','tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',digest(pathlib.Path(library)),'--output',str(out/'occupancy.json')],'occupancy.log')
 run([py,'-m','pytest','-q','--tb=short'],'tests.log',timeout=600)
 for mode in ('tp','pp'):
  run([py,'-u','tools/qualify_qwen_mtp.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--mtp-artifact',str(root/'models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--source',str(root/'models/qwen3.5-27b-bf16'),'--cases',str(root/'venturi-mtp-cases.json'),'--mode',mode,'--output-directory',str(out/f'qualification-{mode}')],f'qualification-{mode}.log')
 # Original one-proposal path: quick matched engine control, not a promoted kernel.
 run([py,'-u','tools/benchmark_qwen_mtp.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--mtp-artifact',str(root/'models/qwen3.5-27b-mtp-q8.ninfer'),'--library',baseline,'--case',str(root/'v100-pp-final-v1/case.json'),'--mode','tp','--concurrency','1','--output-directory',str(out/'original-pair-tile-control')],'original-pair-tile-control.log')
 for mode in ('tp','pp'):
  run([py,'-u','tools/benchmark_qwen_mtp.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--mtp-artifact',str(root/'models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(root/'v100-pp-final-v1/case.json'),'--mode',mode,'--concurrency','1,2,4,8','--output-directory',str(out/f'comparison-{mode}')],f'comparison-{mode}.log',timeout=1200)
 run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-final.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
