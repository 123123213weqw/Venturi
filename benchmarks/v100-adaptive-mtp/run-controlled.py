"""Finite authorized V100 experiment; original production container restored in finally."""
import datetime,hashlib,json,os,pathlib,signal,subprocess,time,urllib.request
ROOT=pathlib.Path('/home/data/wangyue/artifacts/venturi-adaptive-mtp-v1')
REPO=pathlib.Path('/home/data/wangyue/repos/codex-build/Venturi')
SOURCE=REPO/'adaptive-mtp-src-v1'
CID='fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d'
def save(name,data):
 tmp=ROOT/(name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(ROOT/name)
def inspect():return json.loads(subprocess.check_output(['docker','inspect','qwen38-27b']))[0]
def health():return json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4))
def interrupted(*a):raise KeyboardInterrupt('Finite experiment interrupted')
signal.signal(signal.SIGTERM,interrupted)
fd=os.open(ROOT/'controlled.started',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
before=inspect();assert before['Id']==CID and before['State']['Running']
fingerprint=hashlib.sha256(json.dumps({'args':before['Args'],'env':before['Config']['Env'],'mounts':before['Mounts']},sort_keys=True).encode()).hexdigest()
save('production-before.json',{'id':CID,'args':before['Args'],'config_fingerprint':fingerprint,'state':before['State'],'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'authorization':'User explicitly approved temporary pause, followed by restoration'})
rc=1;paused=False
try:
 # Do not cancel an in-flight user generation to obtain the benchmark window.
 idle=0
 for _ in range(90):
  slots=json.load(urllib.request.urlopen('http://127.0.0.1:8000/slots',timeout=5))
  idle=idle+1 if not any(s.get('is_processing') for s in slots) else 0
  if idle>=3:break
  time.sleep(2)
 assert idle>=3,'Production remained busy; pause cancelled'
 paused=True
 subprocess.run(['docker','stop','-t','45','qwen38-27b'],check=True,timeout=75)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(REPO/'v100-tp-diagnostics-v2/python-deps')+':'+str(SOURCE/'src')+':'+str(SOURCE/'tests')+':'+str(SOURCE/'tools'))
 cmd=['/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python',str(SOURCE/'tools/benchmark_qwen_adaptive_mtp.py'),str(REPO/'models/qwen3.5-27b-text-q8.ninfer'),'--mtp-artifact',str(REPO/'models/qwen3.5-27b-mtp-q8.ninfer'),'--library','/home/data/wangyue/artifacts/venturi-native-head-gdn-fixtures-v2/build/68769d5f68c715ef/libventuri_blocks.so','--case',str(REPO/'v100-pp-final-v1/case.json'),'--output-directory',str(ROOT/'full-model'),'--projection-rows','256','--repeats','3']
 save('benchmark-launch.json',{'command':cmd,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_fingerprint':fingerprint})
 with (ROOT/'full-model.log').open('w') as log:
  r=subprocess.run(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 rc=r.returncode;save('benchmark-exit.json',{'returncode':rc})
finally:
 # Also backed by the transient unit's independent ExecStopPost restore hook.
 if paused:
  assert inspect()['Id']==CID,'Container identity changed externally'
  r=subprocess.run(['docker','start','qwen38-27b'],capture_output=True,text=True,timeout=90)
  restored=False
  for _ in range(100):
   try:
    if health()=={'status':'ok'}:restored=True;break
   except Exception:pass
   time.sleep(2)
  after=inspect();afterfp=hashlib.sha256(json.dumps({'args':after['Args'],'env':after['Config']['Env'],'mounts':after['Mounts']},sort_keys=True).encode()).hexdigest()
  save('production-restored.json',{'id':after['Id'],'running':after['State']['Running'],'healthy':restored,'config_fingerprint':afterfp,'configuration_unchanged':afterfp==fingerprint,'start_returncode':r.returncode,'start_stderr':r.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  assert restored and afterfp==fingerprint,'Production restoration needs attention'
raise SystemExit(rc)
