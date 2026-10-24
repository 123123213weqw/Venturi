try:
 import pathlib, subprocess, json, os, time, traceback, signal, urllib.request, hashlib, datetime
 root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-strided-external-depth-v7';src=root/'repos/codex-build/Venturi/strided-src-v2';prior=root/'artifacts/venturi-strided-fixtures-v3'
 py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');deadline=time.monotonic()+1200
 while not (prior/'exit-code.txt').exists():
  process=json.loads((prior/'process.json').read_text());cmdline=pathlib.Path(f"/proc/{process['pid']}/cmdline");assert cmdline.exists() and str(prior/'worker.py').encode() in cmdline.read_bytes(),'Predecessor stopped without terminal evidence';assert time.monotonic()<deadline,'Bounded predecessor wait elapsed';time.sleep(10)
 assert (prior/'exit-code.txt').read_text().strip()=='0'
 library=pathlib.Path((prior/'library.txt').read_text().strip());target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf';port=18951
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
  directory=out/label;directory.mkdir();server=None;samplers=[];logs=[];item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,projection_profile=selected_profile if engine=='venturi' else None,library_sha256=digest(library) if engine=='venturi' else None,direct_primary_output=(library==candidate_library) if engine=='venturi' else None,completed=False)
  try:
   for gpu in [1,0]:
    f=(directory/f'gpu-{gpu}-samples.csv').open('wb');logs.append(f);samplers.append(subprocess.Popen(['nvidia-smi','-i',str(gpu),'--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits','-lms','100'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
   if engine=='llama':
    cmd=[str(bindir/'llama-server'),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-ngl','all','--split-mode',mode,'--tensor-split','1,1','--fit','off','-c','2048','--parallel','8','-b','256','-ub','256','-t','4','-tb','4','-fa','on','-ctk','f16','-ctv','f16','--backend-sampling','--spec-type','draft-mtp' if depth else 'none']
    if depth:cmd+=['--spec-draft-n-max',str(depth),'--spec-draft-backend-sampling']
   else:
    cmd=[py,'tools/serve_native_benchmark.py',str(target),'--library',str(library),'--mode','tp','--port',str(port),'--slots','8','--capacity','256','--ready-file',str(directory/'ready.json')]
    cmd+=['--projection-profile',selected_profile]
    if depth:cmd+=['--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer')]
    if mode=='batch':cmd+=['--batch-target','--batch-window-ms','25']
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



 selected_profile='q8-residual';candidate_library=library
 try:
  gate=root/'artifacts/venturi-strided-final-v6';deadline=time.monotonic()+1800
  while not (gate/'exit-code.txt').exists():
   process=json.loads((gate/'process.json').read_text());cmdline=pathlib.Path(f"/proc/{process['pid']}/cmdline");assert cmdline.exists() and str(gate/'worker.py').encode() in cmdline.read_bytes(),'Pinned predecessor absent without terminal evidence';assert time.monotonic()<deadline,'Bounded predecessor wait elapsed';time.sleep(10)
  assert (gate/'exit-code.txt').read_text().strip()=='0'
  assert digest(bindir/'llama-server')=='e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2'
  control=root/'artifacts/venturi-strided-campaign-v5/campaign-report.json';d=json.loads(control.read_text());assert d['completed'] and d['promoted']
  report.update(scope='Three unprofiled repeats per fresh single-request llama.cpp MTP depth 2/3/4 candidate. Depth1/non-spec controls explicitly reused from immediately preceding matched campaign; best TESTED depth is not a global best-configuration claim. Failed/unsupported depths preserved, never zeros.',binary_sha256=digest(bindir/'llama-server'),reused_control=dict(path=str(control),sha256=digest(control)),source_sha256=d['source_sha256']);save()
  for depth in [2,3,4]:
   try:experiment(f'repeat-llama-tensor-mtp{depth}-c1','llama','tensor',depth,'1',3)
   except Exception:save()
  report['completed']=True;save();rc=0
 except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
 finally:
  run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 (out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
except Exception:
 import pathlib,traceback
 out=pathlib.Path('/home/data/wangyue/artifacts/venturi-strided-external-depth-v7')
 (out/'failure.txt').write_text(traceback.format_exc())
 (out/'exit-code.txt').write_text('1\n')
 raise
