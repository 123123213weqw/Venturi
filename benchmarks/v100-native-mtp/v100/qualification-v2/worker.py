import pathlib,subprocess,json,os,time,traceback
root=pathlib.Path('/home/wzu/codex-build/Venturi');src=root/'v100-mtp-src-v2';out=root/'v100-mtp-qualification-v2'
py='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
def run(cmd,name,timeout=900):
 with (out/name).open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
try:
 assert (root/'v100-mtp-export-v1/exit-code.txt').read_text().strip()=='0'
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(root/'v100-mtp-build')],'library.txt')
 library=(out/'library.txt').read_text().strip();env['VENTURI_SM70_LIBRARY']=library
 run([py,'-m','pytest','-x','-q','tests/test_qwen_mtp.py','tests/test_qwen_native.py','tests/test_qwen_tensor_parallel.py','tests/test_qwen_pipeline.py','--tb=short'],'fixture-tests.log',timeout=300)
 run([py,'-m','pytest','-q','--tb=short'],'tests.log',timeout=600)
 run([py,'-u','tools/qualify_qwen_native.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--expected-sha256','b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8','--blocks-library',library,'--previous-directory',str(root/'greedy-validation-v3/full-model-run1'),'--previous-report-sha256','cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732','--tokenizer-file',str(root/'models/qwen3.5-27b-bf16/tokenizer.json'),'--dtype','fp16','--weight-mode','resident','--pipeline-device','cuda:1','--pipeline-split-layer','32','--output-directory',str(out/'target-qualification')],'target-qualification.log')
 for mode in ('tp','pp'):
  cmd=[py,'-u','tools/qualify_qwen_mtp.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--mtp-artifact',str(root/'models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--source',str(root/'models/qwen3.5-27b-bf16'),'--cases',str(root/'venturi-mtp-cases.json'),'--mode',mode,'--output-directory',str(out/f'mtp-{mode}')]
  if mode=='pp':cmd+=['--skip-block-reference']
  run(cmd,f'mtp-{mode}.log',timeout=900)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
