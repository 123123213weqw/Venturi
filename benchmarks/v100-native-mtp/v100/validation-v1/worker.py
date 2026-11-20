import pathlib,subprocess,json,os,time,traceback
root=pathlib.Path('/home/wzu/codex-build/Venturi');src=root/'v100-mtp-src-v1';out=root/'v100-mtp-validation-v1'
py='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
def run(cmd,name,timeout=900):
 with (out/name).open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
try:
 run([py,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--output-dir',str(root/'v100-mtp-build')],'library.txt')
 library=(out/'library.txt').read_text().strip();env['VENTURI_SM70_LIBRARY']=library
 run([py,'-m','pytest','-x','-q','tests/test_qwen_mtp.py','--tb=short'],'tests.log',timeout=300)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
