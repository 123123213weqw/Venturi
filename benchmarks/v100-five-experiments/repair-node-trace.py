"""Bounded, restored Qwen3.8 five-experiment maintenance window."""
import hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from restore_node import fingerprint,inspect,restore
ROOT=Path('/home/data/wangyue/artifacts/venturi-five-node-repair-v1')
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
 prior=Path('/home/data/wangyue/artifacts/venturi-five-window-v4')
 assert json.loads((prior/'completed.json').read_text())['completed'] and json.loads((prior/'production-restored.json').read_text())['healthy']
 library=Path(json.loads((BUILD/'library.json').read_text())['library']);record=json.loads(library.with_name('build.json').read_text());assert sha(library)==record['library_sha256']
 for f,h in record['source_sha256'].items():assert sha(SOURCE/'src/venturi/ops/csrc'/f)==h
 save('build-record.json',record);save('repair-reason.json',dict(reason='Default nsys graph-level capture omitted graph kernels. Explicit node tracing is required for kernel attribution; never count unobserved graph kernels as eliminated.',prior_reports_preserved=True))
 a=inspect();assert a['State']['Running'] and a['State']['Health']['Status']=='healthy' and api('health').get('status')=='ok'
 before=dict(container_id=a['Id'],canonical_config_fingerprint=fingerprint(a),started_at=a['State']['StartedAt'],counter_policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s],unix=time.time());save('production-before.json',before)
 timer='venturi-five-node-restore-v1'
 subprocess.run(['systemd-run','--user','--unit='+timer,'--on-active=6min',PYTHON,str(CONTROL/'restore_node.py'),'--watchdog'],check=True,timeout=15)
 paused=False
 try:
  idle=0
  for _ in range(60):
   slots=api('slots');idle=idle+1 if slots and not any(s.get('is_processing') for s in slots) else 0
   if idle>=3:break
   time.sleep(2)
  assert idle>=3
  a=inspect();assert a['Id']==before['container_id'] and fingerprint(a)==before['canonical_config_fingerprint']
  paused=True;subprocess.run(['docker','stop','-t','45',a['Id']],check=True,timeout=70);save('production-paused.json',dict(unix=time.time()))
  free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()];assert min(free)>23500
  artifact='/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.8-27b-text-q8-rms-v1.ninfer'
  for mask in [0,15]:
   name=f'timeline-node-{mask}'
   command=[PYTHON,str(CONTROL/'benchmark_model.py'),artifact,'--library',str(library),'--prompts',str(prior/'prompts.json'),'--label','Qwen3.8 native Q8; node-level profiler only, no TPS','--new-tokens','32','--mask',str(mask),'--mode','profile','--output-directory',str(ROOT/name),'--profile-count','8','--counter-start']
   run(name,['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(ROOT/name)]+command,160)
   run(name+'-sqlite',['/usr/local/bin/nsys','export','--type=sqlite','--output',str(ROOT/(name+'.sqlite')),str(ROOT/(name+'.nsys-rep'))],30)
  save('completed.json',dict(completed=True,profiled_only=True,node_level=True,unix=time.time()))
 finally:
  if paused:assert restore()['healthy']
  subprocess.run(['systemctl','--user','stop',timer+'.timer'],check=False,timeout=10)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if ROOT.exists():save('exception.json',dict(error=repr(e),unix=time.time()))
  raise
