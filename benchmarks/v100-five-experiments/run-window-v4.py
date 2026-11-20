"""Bounded, restored Qwen3.8 five-experiment maintenance window."""
import hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from restore_service_v1 import fingerprint,inspect,restore
ROOT=Path('/home/data/wangyue/artifacts/venturi-five-window-v4')
SOURCE=Path('/home/data/wangyue/repos/codex-build/Venturi/five-src-v4')
CONTROL=Path(__file__).parent
BUILD=Path('/home/data/wangyue/artifacts/venturi-five-v2')
EXPORT=Path('/home/data/wangyue/artifacts/venturi-five-mtp-export-v1')
PYTHON='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TOKENIZERS_PARALLELISM='false',
 PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{SOURCE}/src:{SOURCE}/tests:/home/wzu/.local/lib/python3.10/site-packages')
UUIDS=['GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28','GPU-f65db2e8-87bb-c88d-363f-23c79f793548']
METRICS=['gpu__time_duration.sum','sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','dram__bytes_write.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed','l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum','l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum','lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct']
def save(n,j):
 p=ROOT/n;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def api(n):return json.load(urllib.request.urlopen('http://127.0.0.1:8000/'+n,timeout=5))
def interrupted(*_):raise KeyboardInterrupt('Maintenance interrupted: restore service')
def run(name,args,limit,required=True):
 save(name+'.command.json',args)
 with (ROOT/(name+'.log')).open('xb') as f:
  p=subprocess.Popen(args,cwd=SOURCE,env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=limit)
  except BaseException:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=8)
   raise
 save(name+'.exit.json',dict(returncode=rc,unix=time.time()))
 if required:assert rc==0,name+' failed'
 return rc

def main():
 ROOT.mkdir(exist_ok=False);signal.signal(signal.SIGTERM,interrupted)
 for _ in range(180):
  if (BUILD/'completed.json').exists():break
  if subprocess.check_output(['systemctl','--user','show','venturi-five-build-v2','-p','SubState','--value'],text=True).strip() not in ['running','start']:raise RuntimeError('Build/fixture gate failed; production untouched')
  time.sleep(2)
 assert json.loads((BUILD/'completed.json').read_text())['completed']
 library=Path(json.loads((BUILD/'library.json').read_text())['library']);record=json.loads(library.with_name('build.json').read_text());assert sha(library)==record['library_sha256']
 for f,h in record['source_sha256'].items():assert sha(SOURCE/'src/venturi/ops/csrc'/f)==h
 for flag in ['ROW_PAIR','SWIGLU_QUANT','RMS_QUANT']:assert '-DVENTURI_Q8_'+flag+'=1' in record['command']
 save('executed-source-pins.json',{str(p.relative_to(SOURCE)):sha(p) for d in ['src','tests','tools'] for p in (SOURCE/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
 save('build-record.json',record);save('executed-harness-pins.json',{p.name:sha(p) for p in CONTROL.glob('*.py')})
 # Extra replay/EOS/failure tests and CPU source/MTP storage checks leave production running.
 extra=dict(ENV);extra['VENTURI_EXPERIMENT_LIBRARY']=str(library);ENV.update(extra)
 run('extra-fixtures',[PYTHON,'-m','pytest','-xq',str(CONTROL/'test_native_experiments_extra.py')],180)
 for _ in range(240):
  if (EXPORT/'completed.json').exists():break
  if subprocess.check_output(['systemctl','--user','show','venturi-five-mtp-export-v1','-p','SubState','--value'],text=True).strip()!='running':raise RuntimeError('MTP source export failed before production pause')
  time.sleep(2)
 assert (EXPORT/'completed.json').exists(),'Export unfinished; production untouched'
 draft=Path(json.loads((EXPORT/'completed.json').read_text())['artifact']);assert sha(draft)==json.loads((EXPORT/'completed.json').read_text())['sha256']
 # Export verifier is independently implemented; no model expansion on GPU.
 run('verify-mtp-storage',[PYTHON,str(SOURCE/'tools/verify_qwen_q8_export.py'),'/home/wzu/models/Qwen3.8-27B',str(draft),'--export-report',str(EXPORT/'export.json'),'--output',str(ROOT/'mtp-storage-verification.json'),'--component','mtp','--source-manifest','/home/data/wangyue/artifacts/venturi-rms-native-qwen38-export-v2/source-evidence.json'],360)
 artifact=Path('/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.8-27b-text-q8-rms-v1.ninfer')
 assert sha(artifact)=='41f99992b679aef7df2b6c671b76b09820730846c2c95a52406e7004c72f58f0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==UUIDS
 pinned=json.loads((CONTROL/'pinned-prompts.json').read_text());assert len(pinned['prompt_token_ids'])==8 and all(len(x)==16 for x in pinned['prompt_token_ids']);save('prompts.json',pinned)
 a=inspect();assert a['State']['Running'] and a['State']['Health']['Status']=='healthy' and api('health').get('status')=='ok'
 before=dict(container_id=a['Id'],canonical_config_fingerprint=fingerprint(a),started_at=a['State']['StartedAt'],restart_count=a['RestartCount'],counter_policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s],authorization='User approved all five experiments and drained temporary production pause; restore original parameters',unix=time.time());save('production-before.json',before)
 timer='venturi-five-restore-v4'
 subprocess.run(['systemd-run','--user','--unit='+timer,'--on-active=38min',PYTHON,str(CONTROL/'restore_service_v1.py'),'--watchdog'],check=True,timeout=15)
 save('watchdog.json',dict(unit=timer,delay_minutes=38,unix=time.time()))
 paused=False
 try:
  idle=0
  for _ in range(90):
   slots=api('slots');idle=idle+1 if slots and not any(s.get('is_processing') for s in slots) else 0
   if idle>=3:break
   time.sleep(2)
  assert idle>=3,'No idle window'
  a=inspect();assert a['Id']==before['container_id'] and fingerprint(a)==before['canonical_config_fingerprint']
  paused=True;subprocess.run(['docker','stop','-t','45',a['Id']],check=True,timeout=70);save('production-paused.json',dict(unix=time.time(),container_id=a['Id']))
  for _ in range(20):
   free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()]
   if min(free)>23500:break
   time.sleep(2)
  assert min(free)>23500,'Unrelated GPU load; restore without killing it'
  base=[PYTHON,str(CONTROL/'benchmark_model.py'),str(artifact),'--library',str(library),'--prompts',str(ROOT/'prompts.json'),'--label','official Qwen3.8-27B native Q8, not production Uncensored GGUF','--new-tokens','32']
  def command(name,mask,mode,extra=()):return base+['--mask',str(mask),'--mode',mode,'--output-directory',str(ROOT/name)]+list(extra)
  good=[]
  run('qualify-0',command('qualify-0',0,'qualify'),240)
  for mask in [1,2,4,8,15]:
   name='qualify-'+str(mask)
   if run(name,command(name,mask,'qualify',['--reference',str(ROOT/'qualify-0/report.json')]),240,False)==0:good.append(mask)
  save('qualified-masks.json',good);assert good,'All candidates failed; do not score speed'
  reference=ROOT/f'qualify-{good[0]}/report.json'
  for trial in range(3):
   order=[0]+good
   if trial%2:order.reverse()
   for mask in order:
    name=f'timing-{trial}-{mask}';run(name,command(name,mask,'timing',['--reference',str(reference),'--trial',str(trial)]),150)
  save('unprofiled-complete.json',dict(repeats=3,masks=[0]+good,counts=[1,8],unix=time.time()))
  run('mtp',[PYTHON,str(CONTROL/'benchmark_mtp.py'),'--artifact',str(artifact),'--draft',str(draft),'--library',str(library),'--prompts',str(ROOT/'prompts.json'),'--reference',str(reference),'--output',str(ROOT/'mtp')],500,False)
  for mask in [0,15]:
   if mask and mask not in good:continue
   name=f'timeline-{mask}'
   args=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(ROOT/name)]
   if run(name,args+command(name,mask,'profile',['--profile-count','8','--counter-start']),180,False)==0:
    run(name+'-sqlite',['/usr/local/bin/nsys','export','--type=sqlite','--output',str(ROOT/(name+'.sqlite')),str(ROOT/(name+'.nsys-rep'))],40,False)
  for mask,kernel in [(0,'gdn_compact'),(1,'gdn_segments'),(0,'causal_conv'),(1,'conv_segments'),(0,'fp32_weight_linear'),(2,'ab_gates')]:
   if mask and mask not in good:continue
   name=f'counter-{mask}-{kernel}'
   args=['sudo','-n','/usr/bin/env','PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/root','CUDA_VISIBLE_DEVICES=0,1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','PYTHONPATH='+ENV['PYTHONPATH'],
    '/usr/local/cuda/bin/ncu','--clock-control','none','--cache-control','none','--profile-from-start','off','--kernel-name-base','demangled','--kernel-name','regex:.*'+kernel+'.*','--launch-count','1','--metrics',','.join(METRICS),'--csv','--page','raw','--log-file',str(ROOT/(name+'.csv'))]
   run(name,args+command(name,mask,'profile',['--profile-count','8','--counter-start']),180,False)
  save('completed.json',dict(completed=True,qualified=good,default_changed=False,unix=time.time()))
 finally:
  if paused:
   r=restore();assert r['healthy'] and r['config_unchanged']
  else:assert inspect()['Id']==before['container_id'] and api('health').get('status')=='ok'
  subprocess.run(['systemctl','--user','stop',timer+'.timer'],check=False,timeout=10)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if ROOT.exists():save('exception.json',dict(error=repr(e),unix=time.time()))
  raise
