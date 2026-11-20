"""Two matched, quality-qualified implementations, not the broken historical PR."""
from pathlib import Path
import hashlib,json,os,signal,statistics,subprocess,sys,time,urllib.request
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'change/tools'))
from night_blast import blast, qualified_output_ids
from cuda_activity import NvmlProcesses, wait_quiet, ObservedWindow
assert not (R/'evidence/tps-matched-v1.json').exists()
qual=json.loads((R/'evidence/sm70-real-v1-report.json').read_text());assert qual['passed'] and qual['completed']
artifact=json.loads((R/'evidence/artifact.json').read_text());assert qual['artifact_sha256']==artifact['sha256']
vocab=qual['model_dimensions']['vocab_size'];prompt=[(7*i+13)%vocab for i in range(16)];expected=qualified_output_ids(qual,prompt,32)
build=json.loads((R/'evidence/change-sm70-build.json').read_text());source=R/'change';reference=json.loads((R/'evidence/reference-sm70-build.json').read_text())
observer=NvmlProcesses(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
base_env=dict(os.environ,PYTHONPATH=str(source/'src'),PYTHONDONTWRITEBYTECODE='1')
result=dict(scope='Two V100 independent native instances; same pinned Q5 artifact, A16 with FP16 decoded weights, no MTP/prefix reuse; frozen serial conv/GDN original row decode control vs safe-auto coalesced row decode and 2D output copies with batched conv/GDN. Fused Q5 remains unqualified and is excluded; this is not a fusion speedup or a comparison against broken historical code.',artifact=artifact,build=build,reference_build=reference,implementations={},gpu_contamination_policy='Read-only NVML every 50ms; retain/discard contaminated complete cohorts; bounded new timing attempts, never retry failed generation. No observed foreign contexts is not proof of exclusivity between samples.',hardware_counters='not measured; no permissions/monitor changes; outside supervisor samples GPU busy/VRAM/power/clocks/temperature')

# Reuse ONLY the fully completed reference arm, never the interrupted campaign.
# The immutable original JSON and its incomplete top-level status are retained.
prior_root=Path('/home/xs/codex-build/Venturi-pr14-final-20261117-v2')
prior_path=prior_root/'evidence/tps-matched-v1.json'
prior=json.loads(prior_path.read_text())
prior_qual=json.loads((prior_root/'evidence/sm70-real-v1-report.json').read_text())
assert prior_qual['passed'] and prior_qual['completed']
assert prior['artifact']==artifact and prior['reference_build']==reference
assert prior_qual['reference_ops_sha256']==qual['reference_ops_sha256']
assert prior_qual['reference_qwen_sha256']==qual['reference_qwen_sha256']
assert prior_qual['cases_sha256']==qual['cases_sha256']
assert qualified_output_ids(prior_qual,prompt,32)==expected
assert (prior_root/'serve_campaign.py').read_bytes()==(R/'serve_campaign.py').read_bytes()
ref=prior['implementations']['dequant'];assert ref['completed'] and len(ref['runs'])==15
expected_hash=hashlib.sha256(json.dumps(expected).encode()).hexdigest()
for c in (1,4,8,16,32):
 rows=[v for v in ref['runs'] if v['concurrency']==c]
 assert len(rows)==3 and {v['repeat'] for v in rows}=={0,1,2}
 for v in rows:
  assert v['valid'] and not v['errors'] and v['n_predict']==32 and v['prompt_len']==16
  assert v['expected_tokens']==c*32 and v['emitted_tokens']==c*32
  assert v['qualified_output_ids_equal'] and v['output_sha256']==[expected_hash]*c
  assert v['gpu_process_window']['clean'] and not v['gpu_process_window']['errors']
expected_identity=dict(engine='Venturi native',mode='single',slots=32,capacity=256,projection_rows=32,continuous_target=True,prefill_chunk_size=16,prefill_policy='round-robin',projection_profile='fp16',metadata_mode='sealed',mtp_depth=0,target_layer_graph=False)
for identity in ref['server_identity']:
 for k,v in expected_identity.items():assert identity[k]==v,(k,identity[k],v)
result['reference_reuse']=dict(path=str(prior_path),sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),original_campaign_completed=prior.get('completed',False),reused_arm_completed=True,reason='Reuse 15 completed, qualified clean same-workload reference cells; candidate was interrupted before timing. Different time windows explicitly retained; no baseline cell overwritten or relabelled.')
result['implementations']['dequant']=ref

output=R/'evidence/tps-matched-v1.json'
def save():output.write_text(json.dumps(result,indent=2)+'\n')
def procstat(pid):
 raw=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();stat=(Path('/proc')/str(pid)/'status').read_text()
 rss=next(int(x.split()[1]) for x in stat.splitlines()if x.startswith('VmRSS:'))
 return dict(cpu_seconds=(int(raw[11])+int(raw[12]))/os.sysconf('SC_CLK_TCK'),rss_kib=rss)
def get(url):
 with urllib.request.urlopen(url,timeout=5)as r:return json.loads(r.read())
def wait(url,timeout=240):
 start=time.monotonic()
 while time.monotonic()-start<timeout:
  try:return get(url)
  except(OSError,ValueError):time.sleep(1)
 raise RuntimeError('Benchmark listener not ready')
def stop(p):
 if p.poll()is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
for implementation in ('auto',):
 procs=[];run=dict(implementation=implementation,runs=[],warmups=[],server_identity=[],discarded_contaminated_runs=[],failed_generation_runs=[])
 result['implementations'][implementation]=run;save()
 try:
  for ordinal,port in enumerate((19431,19432)):
   env=base_env|{'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'].split(',')[ordinal]}
   ready=R/f'evidence/{implementation}-server{ordinal}-ready.json'
   cmd=[sys.executable,str(R/'serve_campaign.py'),str(R),artifact['path'],'--library',(reference if implementation=='dequant' else build)['ops'],'--qwen-library',(reference if implementation=='dequant' else build)['qwen'],'--mode','single','--port',str(port),'--slots','32','--capacity','256','--ready-file',str(ready),'--continuous-target','--projection-rows','32','--prefill-chunk-size','16','--prefill-policy','round-robin','--metadata-mode','sealed','--linear-implementation',implementation]
   p=subprocess.Popen(cmd,cwd=source,env=env,stdout=(R/f'evidence/{implementation}-server{ordinal}.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True);procs.append(p)
   # Independent restorer can identify and terminate only this exact process group.
   tick=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
   (R/f'private/{implementation}-server{ordinal}-pgid.json').write_text(json.dumps(dict(pid=p.pid,start_ticks=tick)))
  for ordinal,port in enumerate((19431,19432)):wait(f'http://127.0.0.1:{port}/health');run['server_identity'].append(json.loads((R/f'evidence/{implementation}-server{ordinal}-ready.json').read_text()))
  router=subprocess.Popen([sys.executable,str(source/'tools/night_router.py'),'--port','19430','--backend','http://127.0.0.1:19431','--backend','http://127.0.0.1:19432'],cwd=source,env=base_env,stdout=(R/f'evidence/{implementation}-router.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True);procs.append(router)
  tick=(Path('/proc')/str(router.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
  (R/f'private/{implementation}-router-pgid.json').write_text(json.dumps(dict(pid=router.pid,start_ticks=tick)))
  wait('http://127.0.0.1:19430/health')
  # Exact timing prompt output must equal the already-qualified scalar reference.
  # Exact workload was validated before any server process was launched.
  request=dict(prompt=prompt,n_predict=32,id_slot=0,ignore_eos=True,stream=False,cache_prompt=False,temperature=0,repeat_penalty=1)
  for port in (19431,19432):
   req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=180)as r:payload=json.loads(r.read())
   assert payload['tokens']==expected,'Exact timing prompt changed qualified output'
  allowed={p.pid for p in procs[:2]}
  for concurrency in (1,4,8,16,32):
   wait_quiet(observer,allowed)
   warm=blast(19430,concurrency,32,16,vocab,300);assert warm['valid'];run['warmups'].append(warm);save()
   repeat=0
   attempts=0
   while repeat<3:
    attempts+=1;assert attempts<=9, 'Bounded clean-repeat attempts exhausted'
    quiet=wait_quiet(observer,allowed)
    before=[procstat(p.pid)for p in procs[:2]];cpu=time.process_time();start=time.time()
    with ObservedWindow(observer,allowed) as observed:
     row=blast(19430,concurrency,32,16,vocab,300)
    after=[procstat(p.pid)for p in procs[:2]]
    row["qualified_output_ids_equal"] = all(x==hashlib.sha256(json.dumps(expected).encode()).hexdigest() for x in row["output_sha256"])
    if not row["qualified_output_ids_equal"]:
     row["errors"].append({"error":"Qualified timing output trajectory mismatch"});row["valid"]=False;row["aggregate_tps"]=None
    row.update(repeat=repeat,unix=start,client_cpu_seconds=time.process_time()-cpu,server_cpu_seconds=[b['cpu_seconds']-a['cpu_seconds']for a,b in zip(before,after)],server_rss_kib=[b['rss_kib']for b in after])
    row.update(gpu_process_window=observed.report(),quiet_wait=quiet,attempt=attempts)
    if not row['valid']:
     run['failed_generation_runs'].append(row);save();raise RuntimeError('Failed/partial generation; no generation retries')
    if observed.errors:raise RuntimeError('GPU observations unavailable; fail closed')
    if not observed.clean:
     row['contaminated_tps_diagnostic']=row['aggregate_tps'];row['aggregate_tps']=None;row['valid']=False;row['contaminated']=True
     run['discarded_contaminated_runs'].append(row);save();continue
    run['runs'].append(row);save()
    print(json.dumps(dict(implementation=implementation,concurrency=concurrency,repeat=repeat,tps=row['aggregate_tps'])),flush=True)
    repeat+=1
  run['completed']=True;save()
 finally:
  for p in reversed(procs):stop(p)
  time.sleep(3)
result['completed']=True;save()
