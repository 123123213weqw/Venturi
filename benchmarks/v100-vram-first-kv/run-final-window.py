"""Correctness-gated, bounded VRAM-first KV experiments; restores same production CID."""
import hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from restore_final import fingerprint,inspect,restore
ROOT=Path('/home/data/wangyue/artifacts/venturi-kv-tier-window-final-v3');SOURCE=Path('/home/data/wangyue/repos/codex-build/Venturi/kv-tier-src-v3');CONTROL=Path(__file__).parent
BUILD=Path('/home/data/wangyue/artifacts/venturi-kv-tier-v3')
PYTHON='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{SOURCE}/src:{SOURCE}/tests:/home/wzu/.local/lib/python3.10/site-packages')
UUIDS=['GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28','GPU-f65db2e8-87bb-c88d-363f-23c79f793548']
METRICS=['gpu__time_duration.sum','sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','dram__bytes_write.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed','l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum','l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum','lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct']
def save(n,j):
 p=ROOT/n;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def api(n):return json.load(urllib.request.urlopen('http://127.0.0.1:8000/'+n,timeout=5))
def interrupted(*_):raise KeyboardInterrupt('Restore production')
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
 assert json.loads((BUILD/'completed.json').read_text())['completed'],'Build/fixture correctness gate missing; production untouched'
 library=Path(json.loads((BUILD/'library.json').read_text())['library']);record=json.loads(library.with_name('build.json').read_text());assert sha(library)==record['library_sha256']
 for f,h in record['source_sha256'].items():assert sha(SOURCE/'src/venturi/ops/csrc'/f)==h
 local=json.loads((CONTROL/'local-source-pins.json').read_text())
 for f,h in local.items():assert sha(SOURCE/f)==h,f
 save('executed-source-pins.json',local);save('build-record.json',record);save('harness-pins.json',{p.name:sha(p) for p in CONTROL.glob('*.py')})
 ENV['VENTURI_KV_TIER_LIBRARY']=str(library)
 run('fixtures-final',[PYTHON,'-m','pytest','-xq','tests/test_qwen_kv_tier.py','tests/test_qwen_native_cli.py'],120)
 artifact=Path('/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.8-27b-text-q8-rms-v1.ninfer');assert sha(artifact)=='41f99992b679aef7df2b6c671b76b09820730846c2c95a52406e7004c72f58f0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==UUIDS
 a=inspect();assert a['State']['Running'] and a['State']['Health']['Status']=='healthy' and api('health').get('status')=='ok'
 before=dict(container_id=a['Id'],canonical_config_fingerprint=fingerprint(a),started_at=a['State']['StartedAt'],counter_policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s],authorization='User approved drained temporary testing and exact restoration; this task does not deploy native production',unix=time.time());save('production-before.json',before)
 timer='venturi-kv-tier-restore-final-v3';subprocess.run(['systemd-run','--user','--unit='+timer,'--on-active=14min',PYTHON,str(CONTROL/'restore_final.py'),'--watchdog'],check=True,timeout=15)
 paused=False
 try:
  idle=0
  for _ in range(90):
   slots=api('slots');idle=idle+1 if slots and not any(s.get('is_processing') for s in slots) else 0
   if idle>=3:break
   time.sleep(2)
  assert idle>=3,'No idle window';a=inspect();assert a['Id']==before['container_id'] and fingerprint(a)==before['canonical_config_fingerprint']
  paused=True;subprocess.run(['docker','stop','-t','45',a['Id']],check=True,timeout=70);save('production-paused.json',dict(unix=time.time(),container_id=a['Id']))
  for _ in range(20):
   free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()]
   if min(free)>23500:break
   time.sleep(2)
  assert min(free)>23500,'Unrelated GPU load; restore without killing it'
  base=[PYTHON,str(CONTROL/'benchmark_model.py'),str(artifact),'--library',str(library),'--prompts',str(CONTROL/'pinned-prompts.json'),'--label','official Qwen3.8-27B native Q8; not production Uncensored GGUF','--new-tokens','32']
  def command(name,variant,mode,extra=()):return base+['--mode',mode,'--output-directory',str(ROOT/name),'--kv-mode','static' if variant=='static' else 'vram-first','--kv-budget-mib',str(128 if variant=='spill' else 0)]+list(extra)
  prior=Path('/home/data/wangyue/artifacts/venturi-kv-tier-window-v1')
  assert json.loads((prior/'completed.json').read_text())['completed']
  reference=prior/'qualify-static/report.json'
  # Hardening changes only fatal driver-error ownership/guards. Requalify full
  # forced-spill logits, traces, all states and 32-token trajectories on final .so.
  run('qualify-spill-final',command('qualify-spill-final','spill','qualify',['--reference',str(reference)]),260)
  for variant in ['static','auto','spill']:
   n='timing-final-'+variant;run(n,command(n,variant,'timing',['--reference',str(reference),'--trial','0']),200)
  save('unprofiled-complete.json',dict(repeats=1,counts=[1,8],variants=['static','auto','spill'],scope='Final hardened library paired recheck; earlier source has three paired repeats plus 512-context/64K-capacity and separate profilers',unix=time.time()))
  save('completed.json',dict(completed=True,default_changed=False,unix=time.time()))
 finally:
  if paused:r=restore();assert r['healthy'] and r['config_unchanged']
  else:assert inspect()['Id']==before['container_id'] and api('health').get('status')=='ok'
  subprocess.run(['systemctl','--user','stop',timer+'.timer'],check=False,timeout=10)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if ROOT.exists():save('exception.json',dict(error=repr(e),unix=time.time()))
  raise
