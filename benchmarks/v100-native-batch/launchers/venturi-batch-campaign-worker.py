import pathlib, subprocess, json, os, time, traceback, signal, urllib.request, hashlib, datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-native-batch-v3';src=root/'repos/codex-build/Venturi/native-batch-src-v3';prior=root/'artifacts/venturi-native-batch-v2'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');library=pathlib.Path((prior/'library.txt').read_text().strip());target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf';port=18944
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',NCCL_P2P_DISABLE='1',LD_LIBRARY_PATH=str(bindir)+':/usr/local/cuda/lib64:'+str(root/'repos/codex-build/.venvs/b2-v100/lib/python3.10/site-packages/nvidia/nccl/lib'))
report=dict(completed=False,scope='Native cohort batching: exact scalar-state gate, same-ABI16 HTTP baseline/change; screen first, conditional 3-repeat promotion, separate profiling. Reuses original-source GGUF and declared encoding differences from v100-llama-comparison.',experiments=[],physical_gpus=[1,0],utc_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds()),achieved_hardware_counters=None)
def save():(out/'campaign-report.json').write_text(json.dumps(report,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stop(p):
 if p and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run(cmd,path,timeout=600):
 (path.with_suffix(path.suffix+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with path.open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:stop(p);raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {path}')
def experiment(label,engine,mode,depth,counts,repeats):
 directory=out/label;directory.mkdir();server=None;samplers=[];logs=[];item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,completed=False)
 try:
  for gpu in [1,0]:
   f=(directory/f'gpu-{gpu}-samples.csv').open('wb');logs.append(f);samplers.append(subprocess.Popen(['nvidia-smi','-i',str(gpu),'--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits','-lms','100'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
  if engine=='llama':
   cmd=[str(bindir/'llama-server'),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-ngl','all','--split-mode',mode,'--tensor-split','1,1','--fit','off','-c','2048','--parallel','8','-b','256','-ub','256','-t','4','-tb','4','-fa','on','-ctk','f16','-ctv','f16','--backend-sampling','--spec-type','draft-mtp' if depth else 'none']
   if depth:cmd+=['--spec-draft-n-max',str(depth),'--spec-draft-backend-sampling']
  else:
   cmd=[py,'tools/serve_native_benchmark.py',str(target),'--library',str(library),'--mode','tp','--port',str(port),'--slots','8','--capacity','256','--ready-file',str(directory/'ready.json')]
   if mode=='batch':cmd+=['--batch-target','--batch-window-ms','5']
  item['command']=cmd;(directory/'command.json').write_text(json.dumps(cmd,indent=2));f=(directory/'server.log').open('wb');logs.append(f);server=subprocess.Popen(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);item['pid']=server.pid
  for attempt in range(180):
   if server.poll() is not None:raise RuntimeError(f'Server exited during startup: {server.returncode}')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:item['health']=json.load(response);break
   except Exception:time.sleep(1)
  else:raise RuntimeError('Bounded health wait elapsed')
  (directory/'loaded-maps.txt').write_text('\n'.join(s for s in pathlib.Path(f'/proc/{server.pid}/maps').read_text().splitlines() if '.so' in s))
  cmd=[py,'-u','tools/benchmark_completion_api.py','--url',f'http://127.0.0.1:{port}','--server-pid',str(server.pid),'--case',str(case),'--concurrency',counts,'--repeats',str(repeats),'--output-directory',str(directory/'timed'),'--expect-pinned-ids']
  run(cmd,directory/'benchmark.log',timeout=600);measured=json.loads((directory/'timed/report.json').read_text());assert measured['completed']
  if engine=='venturi' and mode=='batch':
   for w in measured['waves']:
    assert {r['response']['batch_request_count'] for r in w['requests']}=={w['request_count']},'Admission split a fixed cohort'
    assert len({r['response']['batch_cohort_id'] for r in w['requests']})==1
  item.update(completed=True,summary=measured['summary'],stable_outputs=measured['all_outputs_equal_first_wave'])
 except BaseException:item['failure']=traceback.format_exc();(directory/'failure.txt').write_text(item['failure']);raise
 finally:
  stop(server)
  for p in samplers:stop(p)
  for f in logs:f.close()
  report['experiments'].append(item);save()
 print(json.dumps(item),flush=True);return item
try:
 assert (prior/'exit-code.txt').read_text().strip()=='0'
 proof=json.loads((prior/'library-reuse-proof.json').read_text());assert digest(library)==proof['library_sha256']
 for path,sha in proof['native_source_sha256'].items():assert digest(src/path)==sha
 report.update(library=str(library),library_sha256=digest(library),source_hashes={str(p.relative_to(src)):digest(p) for d in ('src','tests','tools') for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},previous_gpu_fixture_report=str(prior),llama_binary_hash=digest(bindir/'llama-server'),llama_build_info=(bindir.parent/'BUILD_INFO').read_text(),gguf=str(gguf),gguf_sha256_reused='eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513')
 save();run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-before.txt')
 run([py,'tools/qualify_qwen_batch.py',str(target),'--library',str(library),'--cases',str(cases),'--output-directory',str(out/'qualification')],out/'qualification.log',timeout=300)
 baseline=experiment('screen-scalar','venturi','scalar',0,'1,8',1);candidate=experiment('screen-batch','venturi','batch',0,'1,8',1)
 gains={str(b['request_count']):c['aggregate_complete_request_output_tps']/b['aggregate_complete_request_output_tps'] for b,c in zip(baseline['summary'],candidate['summary'])};report['screen_gains']=gains;save()
 if gains['8']>1.1 and gains['1']>=0.9:
  experiment('promotion-scalar','venturi','scalar',0,'1,2,4,8',3)
  experiment('promotion-batch','venturi','batch',0,'1,2,4,8',3)
  experiment('external-tensor-off','llama','tensor',0,'1,2,4,8',3)
  experiment('external-layer-mtp1','llama','layer',1,'1,2,4,8',3)
 report['promoted_to_repeated_matrix']=gains['8']>1.1 and gains['1']>=0.9;save()
 nsys='/opt/nvidia/nsight-systems/2025.3.1/bin/nsys'
 if not pathlib.Path(nsys).exists():nsys=subprocess.check_output(['which','nsys'],text=True).strip()
 traces=[]
 for mode in ['scalar','batch']:
  prefix=out/f'profile-{mode}';cmd=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_qwen_batch.py',str(target),'--library',str(library),'--case',str(case),'--mode',mode,'--output',str(out/f'profile-{mode}-metadata.json')]
  run(cmd,out/f'profile-{mode}.log',timeout=360)
  run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],out/f'profile-{mode}-export.log',timeout=180)
  run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/f'profile-{mode}-summary.json')],out/f'profile-{mode}-summary.log',timeout=120)
  traces+=['--trace',mode+'='+str(prefix)+'.sqlite']
 run([py,'tools/inspect_cuda_occupancy.py',str(library),'--expected-sha256',digest(library),'--devices','0,1',*traces,'--output',str(out/'theoretical-occupancy.json')],out/'occupancy.log',timeout=120)
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
