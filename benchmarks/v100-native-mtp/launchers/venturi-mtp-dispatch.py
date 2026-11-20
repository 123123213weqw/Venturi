import subprocess
worker=r'''import pathlib,subprocess,json,os,time,traceback
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
'''
export=r'''import pathlib,subprocess,json,os,time,traceback
root=pathlib.Path('/home/wzu/codex-build/Venturi');out=root/'v100-mtp-export-v1';src=root/'v100-mtp-src-v1'
code="import torch,json;torch.set_num_threads(4);from venturi.qwen_export import export_qwen_mtp_q8;r=export_qwen_mtp_q8('/home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16','/home/wzu/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer');print(json.dumps(r,indent=2))"
env=os.environ.copy();env.update(PYTHONPATH=str(src/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
try:
 with (out/'export.json').open('w') as log, (out/'export.log').open('w') as err:subprocess.run(['/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python','-u','-c',code],cwd=src,env=env,stdout=log,stderr=err,check=True,timeout=900)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
code=f'''import pathlib,subprocess,time,json
root=pathlib.Path('/home/wzu/codex-build/Venturi')
for name,worker in [('v100-mtp-validation-v1',{worker!r}),('v100-mtp-export-v1',{export!r})]:
 p=root/name;p.mkdir();(p/'worker.py').write_text(worker)
 with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 r=dict(pid=proc.pid,started_unix=time.time());(p/'process.json').write_text(json.dumps(r,indent=2));print(name,json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
