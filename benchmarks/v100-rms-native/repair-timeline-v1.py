"""Separate profiler-only repair after the scored campaign restores production.

The first NVTX capture filter produced no report. Capture the complete process
and restrict attribution offline to the known post-first NVTX range instead.
No profiled wall time is a throughput score.
"""
import importlib.util,json,os,signal,subprocess,time
from pathlib import Path
CONTROL=Path('/home/data/wangyue/repos/codex-build/Venturi/rms-native-window-control-v4')
spec=importlib.util.spec_from_file_location('window_helpers',CONTROL/'run-model-window-v4.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
ROOT=Path('/home/data/wangyue/artifacts/venturi-rms-timeline-repair-v1')
ORIGINAL=c.ROOT;c.ROOT=ROOT
import restore_service_v1 as restoration
restoration.ROOT=ROOT

def main():
 for _ in range(200):
  if (ORIGINAL/'production-restored.json').exists() and (ORIGINAL/'benchmark-complete.json').exists():break
  time.sleep(3)
 else:raise TimeoutError('Initial campaign not restored/completed; no second maintenance window')
 assert json.loads((ORIGINAL/'full-model-correctness-passed.json').read_text())['exact_bytes']
 ROOT.mkdir(exist_ok=False);signal.signal(signal.SIGTERM,c.interrupted)
 before=json.loads((ORIGINAL/'production-before.json').read_text());a=c.inspect()
 assert a['Id']==before['container_id'] and c.fingerprint(a)==before['canonical_config_fingerprint'] and c.api('health').get('status')=='ok'
 before.update(started_at=a['State']['StartedAt'],unix=time.time());c.save('production-before.json',before)
 # Small dedicated restore wrapper; no global service or profiler-policy edits.
 wrapper=Path(__file__).with_name('restore_timeline_repair_v1.py')
 timer='venturi-rms-timeline-repair-restore-v1'
 subprocess.run(['systemd-run','--user','--unit='+timer,'--on-active=12min',c.PYTHON,str(wrapper),'--watchdog'],check=True,timeout=15)
 paused=False
 try:
  idle=0
  for _ in range(60):
   slots=c.api('slots');idle=idle+1 if slots and not any(s.get('is_processing') for s in slots) else 0
   if idle>=3:break
   time.sleep(2)
  assert idle>=3
  a=c.inspect();assert a['Id']==before['container_id'] and c.fingerprint(a)==before['canonical_config_fingerprint']
  paused=True;subprocess.run(['docker','stop','-t','45',a['Id']],check=True,timeout=70);c.save('production-paused.json',dict(unix=time.time(),container_id=a['Id']))
  free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()];assert min(free)>23500
  libs=json.loads((c.BUILD/'libraries.json').read_text());artifact=json.loads((c.EXPORT/'export-report.json').read_text())['output']
  for arm in ['off','on']:
   name='timeline-'+arm
   args=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--output',str(ROOT/name),c.PYTHON,str(CONTROL/'benchmark_native_rms_model.py'),artifact,
    '--prompts',str(ORIGINAL/'prompts.json'),'--label','Qwen3.8 native Q8 RMS '+arm+'; profiler-only repair, not a TPS score','--library',libs[arm],'--mode','profile','--profile-count','8','--output-directory',str(ROOT/name)]
   c.run(name,args,180)
   assert (ROOT/(name+'.nsys-rep')).is_file()
   c.run(name+'-sqlite',['/usr/local/bin/nsys','export','--type=sqlite','--output',str(ROOT/(name+'.sqlite')),str(ROOT/(name+'.nsys-rep'))],60)
  c.save('completed.json',dict(completed=True,profiled=True,scored_as_tps=False,reason=__doc__,unix=time.time()))
 finally:
  if paused:assert restoration.restore()['healthy']
  subprocess.run(['systemctl','--user','stop',timer+'.timer'],check=True,timeout=15)

if __name__=='__main__':main()
