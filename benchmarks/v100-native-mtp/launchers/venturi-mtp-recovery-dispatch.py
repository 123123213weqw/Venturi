import subprocess
worker=r'''import pathlib,subprocess,json,os,time,hashlib,traceback
root=pathlib.Path('/home/wzu/codex-build/Venturi');src=root/'v100-mtp-src-v3';out=root/'v100-mtp-qualification-v3'
py='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_MTP_TEST='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
def run(cmd,name,timeout=900):
 with (out/name).open('w') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 library=pathlib.Path((root/'v100-mtp-qualification-v2/library.txt').read_text().strip());build=json.loads((library.parent/'build.json').read_text())
 assert digest(library)==build['library_sha256']
 assert all(digest(src/'src/venturi/ops/csrc'/f)==h for f,h in build['source_sha256'].items())
 prior=root/'v100-mtp-qualification-v2/target-qualification/report.json';r=json.loads(prior.read_text());assert r['native_full_runtime_case_passed']
 (out/'prerequisites.json').write_text(json.dumps(dict(recovery='First trained MTP block diagnostic stopped on a missing centered_cosine field in the metrics helper. Added explicit cosine computation; gates remain relative L2 <= .02, centered cosine >= .999. Native source/library unchanged. No speed runs had started.',target_qualification_report_sha256=digest(prior),library_sha256=digest(library),source_sha256=build['source_sha256']),indent=2))
 (out/'library.txt').write_text(str(library)+'\n');env['VENTURI_SM70_LIBRARY']=str(library)
 run([py,'-m','pytest','-q','--tb=short'],'tests.log',timeout=600)
 for mode in ('tp','pp'):
  cmd=[py,'-u','tools/qualify_qwen_mtp.py',str(root/'models/qwen3.5-27b-text-q8.ninfer'),'--mtp-artifact',str(root/'models/qwen3.5-27b-mtp-q8.ninfer'),'--library',str(library),'--source',str(root/'models/qwen3.5-27b-bf16'),'--cases',str(root/'venturi-mtp-cases.json'),'--mode',mode,'--output-directory',str(out/f'mtp-{mode}')]
  if mode=='pp':cmd+=['--skip-block-reference']
  run(cmd,f'mtp-{mode}.log',timeout=900)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
code=f'''import pathlib,subprocess,time,json
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-mtp-qualification-v3');p.mkdir();(p/'worker.py').write_text({worker!r})
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
