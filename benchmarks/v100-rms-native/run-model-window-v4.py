"""Authorized Qwen3.8 OFF/ON exact-correctness gate, paired timing and separate diagnostics."""
import hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from restore_service_v1 import fingerprint,inspect,restore
ROOT=Path('/home/data/wangyue/artifacts/venturi-rms-model-window-v4')
SOURCE=Path('/home/data/wangyue/repos/codex-build/Venturi/rms-native-src-v7')
BUILD=Path('/home/data/wangyue/artifacts/venturi-rms-native-v1')
EXPORT=Path('/home/data/wangyue/artifacts/venturi-rms-native-qwen38-export-v2')
PYTHON='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
 PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{SOURCE}/src:{SOURCE}/tests:/home/wzu/.local/lib/python3.10/site-packages')
EXPECTED_UUIDS=['GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28','GPU-f65db2e8-87bb-c88d-363f-23c79f793548']
METRICS=['gpu__time_duration.sum','sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','dram__bytes_write.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed','l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum','l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum','lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct']
def save(n,j):
 p=ROOT/n;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def api(n):return json.load(urllib.request.urlopen('http://127.0.0.1:8000/'+n,timeout=5))
def interrupted(*_):raise KeyboardInterrupt('Maintenance interrupted; restore exact service')
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
 if required:assert rc==0,name+' failed; speed gate remains closed'
 return rc

def main():
 for _ in range(600):
  if (BUILD/'completed.json').exists() and Path('/home/data/wangyue/artifacts/venturi-rms-native-qwen38-storage-v1/completed.json').exists():break
  time.sleep(3)
 else:raise TimeoutError('Prerequisites not ready; production untouched')
 ROOT.mkdir(exist_ok=False)
 signal.signal(signal.SIGTERM,interrupted)
 # CPU export/build complete before any maintenance or GPU allocation.
 assert json.loads((BUILD/'completed.json').read_text())['completed']
 assert json.loads((EXPORT/'completed.json').read_text())['completed']
 assert json.loads(Path('/home/data/wangyue/artifacts/venturi-rms-native-qwen38-storage-v1/completed.json').read_text())['completed']
 libs=json.loads((BUILD/'libraries.json').read_text());records={}
 for name in ['off','on']:
  p=Path(libs[name]);record=json.loads(p.with_name('build.json').read_text());assert sha(p)==record['library_sha256']
  for f,h in record['source_sha256'].items():assert sha(SOURCE/'src/venturi/ops/csrc'/f)==h
  records[name]=record
 assert records['off']['source_sha256']==records['on']['source_sha256']
 def comparable(cmd):return [x for x in cmd if not x.startswith('-DVENTURI_Q8_RMS_QUANT=')][:-1]
 assert comparable(records['off']['command'])==comparable(records['on']['command'])
 assert '-DVENTURI_Q8_RMS_QUANT=0' in records['off']['command'] and '-DVENTURI_Q8_RMS_QUANT=1' in records['on']['command']
 save('build-records.json',records)
 save('executed-harness-sha256.json',{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
 export=json.loads((EXPORT/'export-report.json').read_text());artifact=Path(export['output']);assert artifact.stat().st_size==export['file_bytes']
 save('checkpoint-pin.json',export)
 uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines();assert uuids==EXPECTED_UUIDS
 # Tokenize on the client using the exact official tokenizer, transfer only pinned IDs.
 pinned=json.loads(Path(__file__).with_name('pinned-prompts.json').read_text())
 pin=json.loads((EXPORT/'source-evidence.json').read_text())
 assert pinned['tokenizer_sha256']==next(f['actual_sha256'] for f in pin['files'] if f['name']=='tokenizer.json')
 assert len(pinned['prompt_token_ids'])==8 and all(len(x)==16 for x in pinned['prompt_token_ids']) and pinned['chat_template_applied'] is False
 save('prompts.json',pinned)
 a=inspect();assert a['State']['Running'] and a['State']['Health']['Status']=='healthy' and api('health').get('status')=='ok'
 before=dict(container_id=a['Id'],canonical_config_fingerprint=fingerprint(a),started_at=a['State']['StartedAt'],restart_count=a['RestartCount'],
  counter_policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s],
  authorization='User approved a temporary drained production pause; Qwen3.8 compatibility/correctness first, then restore original parameters.',unix=time.time())
 save('production-before.json',before)
 # Independent timer runs even if client SSH or controller disappears.
 timer='venturi-rms-model-restore-v4'
 result=subprocess.run(['systemd-run','--user','--unit='+timer,'--on-active=27min',PYTHON,str(Path(__file__).with_name('restore_service_v1.py')),'--watchdog'],capture_output=True,text=True,timeout=15)
 assert result.returncode==0,(result.stdout,result.stderr)
 save('watchdog.json',dict(unit=timer,delay_minutes=27,armed_unix=time.time()))
 paused=False
 try:
  idle=0
  for _ in range(60):
   slots=api('slots');idle=idle+1 if slots and not any(s.get('is_processing') for s in slots) else 0
   if idle>=3:break
   time.sleep(2)
  assert idle>=3,'No idle window; maintenance cancelled'
  a=inspect();assert a['Id']==before['container_id'] and fingerprint(a)==before['canonical_config_fingerprint']
  paused=True;subprocess.run(['docker','stop','-t','45',a['Id']],check=True,timeout=70)
  save('production-paused.json',dict(container_id=a['Id'],unix=time.time()))
  for _ in range(20):
   free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()]
   if min(free)>23500:break
   time.sleep(2)
  assert min(free)>23500,'Unrelated GPU workload remains; restore service without killing it'
  base=[PYTHON,str(Path(__file__).with_name('benchmark_native_rms_model.py')),str(artifact),'--prompts',str(ROOT/'prompts.json'),'--label','Qwen3.8-27B official BF16 source -> native Q8 group32 FP16 scale; not production Uncensored GGUF','--new-tokens','32']
  def command(name,which,mode,extra=()):return base+['--library',libs[which],'--mode',mode,'--output-directory',str(ROOT/name)]+list(extra)
  run('qualify-off',command('qualify-off','off','qualify'),240)
  reference=ROOT/'qualify-off/report.json'
  run('qualify-on',command('qualify-on','on','qualify',['--reference',str(reference)]),240)
  gate=ROOT/'qualify-on/report.json';assert json.loads(gate.read_text())['exact_reference_bytes_equal']
  save('full-model-correctness-passed.json',dict(exact_bytes=True,baseline_sha256=sha(reference),candidate_sha256=sha(gate),unix=time.time()))
  for trial,order in enumerate([['off','on'],['on','off'],['off','on']]):
   for which in order:
    name=f'timing-{trial}-{which}'
    run(name,command(name,which,'timing',['--reference',str(gate),'--trial',str(trial)]),150)
  save('unprofiled-complete.json',dict(paired_trials=3,counts=[1,8],unix=time.time()))
  # Profilers are separate from all scored timings; one post-first C8 window.
  for which in ['off','on']:
   name='timeline-'+which
   args=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=nvtx','--nvtx-capture=rms_native_post_first','--capture-range-end=stop','--output',str(ROOT/name)]
   rc=run(name,args+command(name,which,'profile',['--profile-count','8']),180,False)
   if rc==0:
    run(name+'-sqlite',['/usr/local/bin/nsys','export','--type=sqlite','--output',str(ROOT/(name+'.sqlite')),str(ROOT/(name+'.nsys-rep'))],40,False)
  # Privileged only for these bounded NCU commands; do not change driver policy/clocks.
  for which,kernel,skip in [('off','regex:.*rms_norm.*',1),('on','regex:_ZN3vrq5fused.*',0)]:
   name='counter-'+which
   args=['sudo','-n','/usr/bin/env','PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/root','CUDA_VISIBLE_DEVICES=0,1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','PYTHONPATH='+ENV['PYTHONPATH'],
    '/usr/local/cuda/bin/ncu','--clock-control','none','--cache-control','none','--profile-from-start','off','--kernel-name-base','mangled','--kernel-name',kernel,'--launch-skip',str(skip),'--launch-count','1','--metrics',','.join(METRICS),'--csv','--page','raw','--log-file',str(ROOT/(name+'.csv'))]
   run(name,args+command(name,which,'profile',['--profile-count','8','--counter-start']),180,False)
  save('benchmark-complete.json',dict(completed=True,production_candidate_deployed=False,unix=time.time()))
 finally:
  if paused:
   r=restore();assert r['healthy'] and r['config_unchanged']
  else:
   assert inspect()['Id']==before['container_id'] and api('health').get('status')=='ok'
  subprocess.run(['systemctl','--user','stop',timer+'.timer'],check=True,timeout=15)

if __name__=='__main__':main()
