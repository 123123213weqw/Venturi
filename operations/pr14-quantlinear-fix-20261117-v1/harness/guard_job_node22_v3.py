"""Bounded GPU job on freshly idle UUIDs; never stops a foreign process."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
R=Path('/home/xs/codex-build/Venturi-pr14-coalesced-20261117');E=R/'evidence'
cfg=json.loads(Path(sys.argv[1]).read_text());name=cfg['name'];assert name.replace('-','').replace('_','').isalnum()
assert not (E/(name+'-launch.json')).exists(), 'No duplicate experiment'
gpus=cfg['gpus'];assert 1<=len(gpus)<=8 and len(set(gpus))==len(gpus)
assert os.getuid()==1001 and os.uname().nodename=='root1'
lease_dir=R/'leases';lease_dir.mkdir(mode=0o700,exist_ok=True)
def query(flag,fields):return subprocess.check_output(['nvidia-smi',flag+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=10)
def apps():
 out=[]
 for row in query('--query-compute-apps','pid,gpu_uuid,used_memory').strip().splitlines():
  pid,gpu,mem=map(str.strip,row.split(','))
  if gpu in gpus:out.append((int(pid),gpu,mem))
 return out
def small_same_user(row):
 try:return Path('/proc',str(row[0])).stat().st_uid==os.getuid() and Path('/proc',str(row[0]),'comm').read_text().strip()!='llama-server'
 except FileNotFoundError:return True
 except (OSError,ValueError):return False
locks=[]
for gpu in sorted(gpus):
 f=(lease_dir/(gpu+'.lock')).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert not apps(),'Selected cards occupied; do not evict owners'
for row in query('--query-gpu','uuid,memory.used').splitlines():
 gpu,mem=map(str.strip,row.split(','))
 if gpu in gpus:assert int(mem)<64,(gpu,mem)
owned={}
def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();assert p.stat().st_uid==os.getuid()
 return int(s[1]),s[19]
def belongs(pid,root):
 original=pid;seen=set()
 try:
  while pid>1 and pid not in seen:
   seen.add(pid);parent,tick=identity(pid)
   if pid==root:owned[original]=identity(original)[1];return True
   pid=parent
 except(OSError,AssertionError,ValueError):pass
 return False
limit=cfg['seconds'];assert type(limit)is int and 1<=limit<=14400
source=R/cfg['source'];assert source.is_dir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(source/'src')+':'+str(R/'python-deps'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TMPDIR=str(R/'tmp'))
build=json.loads((E/'native-active-build.json').read_text())
env.update(VENTURI_OPS_LIBRARY=build['ops'],VENTURI_QWEN_LIBRARY=build['qwen'])
env['LD_LIBRARY_PATH']=str(Path(build['ops']).parent)+':/usr/local/cuda-12.8/lib64:'+env.get('LD_LIBRARY_PATH','')
env.update(cfg.get('env',{}))
libraries={k:dict(path=env[k],sha256=hashlib.sha256(Path(env[k]).read_bytes()).hexdigest())for k in ('VENTURI_OPS_LIBRARY','VENTURI_QWEN_LIBRARY')}
frozen={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()for root in ('src','tools','tests')for p in sorted((source/root).rglob('*'))if p.is_file()and p.suffix in ('.py','.cu','.cuh','.h','.json')and '__pycache__'not in p.parts}
(E/(name+'-source.json')).write_text(json.dumps(frozen,indent=2))
cmd=['timeout','--signal=TERM','--kill-after=30',str(limit)]+cfg['command']
p=None;reason='normal';began=time.time()
def stop(sig,frame):raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 p=subprocess.Popen(cmd,cwd=source,env=env,stdout=(E/(name+'.log')).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
 (E/(name+'-launch.json')).write_text(json.dumps(dict(pid=p.pid,start_ticks=identity(p.pid)[1],started_unix=began,deadline_unix=began+limit,command=cmd,gpus=gpus,source=str(source),libraries=libraries,source_manifest_sha256=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()),indent=2));print(name,p.pid,flush=True)
 with (E/(name+'-gpu.jsonl')).open('x')as samples:
  while p.poll()is None:
   if time.time()-began>limit+5:reason='deadline';break
   foreign=[x for x in apps()if not belongs(x[0],p.pid)]
   if foreign:
    tolerated=cfg.get('foreign_policy')=='record-same-user' and all(small_same_user(x) for x in foreign)
    if not tolerated:reason='foreign_arrival:'+repr(foreign);break
   data=query('--query-gpu','index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu')
   selected=[row for row in data.splitlines()if row.split(',')[1].strip()in gpus]
   samples.write(json.dumps(dict(timestamp=time.time(),data=selected,foreign_apps=foreign))+'\n');samples.flush();time.sleep(1)
finally:
 if p is not None and p.poll()is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
 for sig in (signal.SIGTERM,signal.SIGKILL):
  for pid,tick in owned.items():
   try:
    if identity(pid)[1]==tick:os.kill(pid,sig)
   except(OSError,AssertionError,ValueError):pass
  time.sleep(.3)
 remaining=[]
 for row in apps():
  try:
   if row[0]in owned and identity(row[0])[1]==owned[row[0]]:remaining.append(row)
  except(OSError,AssertionError,ValueError):pass
 result=dict(reason=reason,exit_code=None if p is None else p.returncode,finished_unix=time.time(),remaining_own_apps=remaining,scope='No foreign job/service/driver/monitor changes; private leases released')
 (E/(name+'-final.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 for f in locks:f.close()
if reason!='normal' or p is None or p.returncode!=0 or remaining:raise SystemExit(1)
