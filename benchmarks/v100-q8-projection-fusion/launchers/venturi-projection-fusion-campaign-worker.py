import pathlib, subprocess, json, os, time, traceback, signal, urllib.request, hashlib, datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-projection-fusion-campaign-v4';src=root/'repos/codex-build/Venturi/projection-fusion-src-v3';prior=root/'artifacts/venturi-projection-fusion-model-v3'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');deadline=time.monotonic()+1200
while not (prior/'exit-code.txt').exists():
 process=json.loads((prior/'process.json').read_text());cmdline=pathlib.Path(f"/proc/{process['pid']}/cmdline");assert cmdline.exists() and str(prior/'worker.py').encode() in cmdline.read_bytes(),'Predecessor stopped without terminal evidence';assert time.monotonic()<deadline,'Bounded predecessor wait elapsed';time.sleep(10)
assert (prior/'exit-code.txt').read_text().strip()=='0'
library=pathlib.Path((prior/'library.txt').read_text().strip());target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf';port=18953
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
 directory=out/label;directory.mkdir();server=None;samplers=[];logs=[];item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,projection_profile=selected_profile if engine=='venturi' else None,library_sha256=digest(library) if engine=='venturi' else None,projection_group_mode=(2 if library==candidate_library else 1 if library==grouped_library else 0) if engine=='venturi' else None,completed=False)
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



candidate_library=library;baseline_library=root/'artifacts/venturi-tiled-staging-model-v3/build/398e4f4a1dd037ea/libventuri_blocks.so';grouped_library=root/'artifacts/venturi-projection-fusion-v1/build/b08a62aa2ac57cc2/libventuri_blocks.so';selected_profile='q8-residual'
try:
 assert digest(baseline_library)=='a6599e636cc0c018261310de7d83e1e8dd41d6edcc635f9161402a89f7775a3f'
 assert digest(grouped_library)=='6698f47bae3405279102890818782c992e8179aec966af8b5a08941160bb28b1'
 report['qualification_gates']=[]
 for name in ['venturi-projection-fusion-model-v2','venturi-projection-fusion-model-v3']:
  gate=root/'artifacts'/name;assert (gate/'exit-code.txt').read_text().strip()=='0';q=json.loads((gate/'qualification/report.json').read_text());assert q['completed'] and q['executed_candidate_profiles']==['q8-residual'] and q['profiles']['q8-residual']['same_profile_bytes_equal']
  mtp=json.loads((gate/'mtp-qualification/report.json').read_text());assert mtp['completed'] and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] for c in mtp['cases'])
  report['qualification_gates'].append(dict(path=str(gate),qualification_report_sha256=digest(gate/'qualification/report.json'),mtp_report_sha256=digest(gate/'mtp-qualification/report.json'),candidate_library_sha256=q['candidate_library_sha256'],unaffected_fp16_reference_reused=True,draft_cpu_reference_reused=True))
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Same-input projection grouping: mode0 pinned old fences, mode1 grouped transport with original kernels, mode2 grouped transport and fused complete row-dot kernels. Identical Q8-residual arithmetic. Fresh trained state gates precede matched screens and conditional three-repeat promotion. Prior strongest external controls explicitly reused, not selected downward.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),grouped_library_sha256=digest(grouped_library),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},tensor_core_model_dispatch=False);save()
 library=baseline_library;a=experiment('screen-control','venturi','batch',0,'1,2,8',1)
 library=grouped_library;b=experiment('screen-grouped','venturi','batch',0,'1,2,8',1)
 library=candidate_library;c=experiment('screen-fused','venturi','batch',0,'1,2,8',1)
 report['screen_gains']={v['label']:{str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],v['summary'])} for v in [b,c]};save()
 library=baseline_library;d=experiment('screen-mtp-control','venturi','scalar',1,'1',1)
 library=grouped_library;e=experiment('screen-mtp-grouped','venturi','scalar',1,'1',1)
 library=candidate_library;f=experiment('screen-mtp-fused','venturi','scalar',1,'1',1)
 report['mtp_screen_gains']={v['label']:v['summary'][0]['aggregate_complete_request_output_tps']/d['summary'][0]['aggregate_complete_request_output_tps'] for v in [e,f]}
 qualified=[]
 for tag,lib,v in [('grouped',grouped_library,b),('fused',candidate_library,c)]:
  g=report['screen_gains'][v['label']];mtp_gain=report['mtp_screen_gains']['screen-mtp-'+tag]
  if g['1']>1.015 and min(g['2'],g['8'],mtp_gain)>.99:qualified.append((g['1']*g['8']*mtp_gain,tag,lib))
 report['promotion_rule']='C1 >1.5%; no >1% C2/C8/MTP screen regression. Highest product of C1*C8*MTP HTTP ratios wins. Internal screening only; the >=5% external acceptance goal is unchanged.'
 if qualified:
  _,tag,chosen=max(qualified);report.update(promoted=True,promoted_variant=tag,promoted_library_sha256=digest(chosen));save()
  library=baseline_library;experiment('repeat-control','venturi','batch',0,'1,2,4,8',3)
  library=chosen;experiment('repeat-'+tag,'venturi','batch',0,'1,2,4,8',3)
  library=baseline_library;experiment('repeat-mtp-control','venturi','scalar',1,'1',3)
  library=chosen;experiment('repeat-mtp-'+tag,'venturi','scalar',1,'1',3)
 else:report['promoted']=False;save()
 report['reused_external_controls']=[]
 for path in [root/'artifacts/venturi-wmma-campaign-v5/campaign-report.json',root/'artifacts/venturi-strided-external-depth-v9/campaign-report.json']:
  external=json.loads(path.read_text());assert external['completed'];report['reused_external_controls'].append(dict(path=str(path),sha256=digest(path)))
 report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
