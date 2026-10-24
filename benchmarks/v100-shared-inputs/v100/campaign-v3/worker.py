import pathlib, subprocess, json, os, time, traceback, signal, urllib.request, hashlib, datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-shared-input-campaign-v3';src=root/'repos/codex-build/Venturi/shared-input-src-v2';prior=root/'artifacts/venturi-shared-input-fixtures-v1'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');library=pathlib.Path((prior/'library.txt').read_text().strip());target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf';port=18948
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
 directory=out/label;directory.mkdir();server=None;samplers=[];logs=[];item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,projection_profile=selected_profile if engine=='venturi' else None,library_sha256=digest(library) if engine=='venturi' else None,shared_inputs=(library==candidate_library) if engine=='venturi' else None,completed=False)
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


candidate_library=library;baseline_library=root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so';selected_profile='q8-residual'
try:
 gate=root/'artifacts/venturi-shared-input-qualification-v2';assert (gate/'exit-code.txt').read_text().strip()=='0';qualified=json.loads((gate/'qualification/report.json').read_text());assert qualified['completed'] and all(q['same_profile_bytes_equal'] for q in qualified['profiles'].values())
 build=json.loads(candidate_library.with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==sha
 report.update(scope='Explicit same-input projection groups: unchanged arithmetic within each profile; retained cross-device output-reuse fences. Trained old/new byte gates before unprofiled 1/8 screening; conditional three-repeat matrix; separate diagnostics. Original/residual arithmetic are not interchangeable.',baseline_library_sha256=digest(baseline_library),candidate_library_sha256=digest(candidate_library),qualification_report_sha256=digest(gate/'qualification/report.json'),source_sha256={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts});save()
 run([py,'tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',str(candidate_library),'--source',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual','--skip-block-reference','--output-directory',str(out/'mtp-qualification')],out/'mtp-qualification.log',timeout=300)
 gains={}
 for profile in ['q8-residual','fp16']:
  selected_profile=profile;library=baseline_library;a=experiment('screen-unshared-'+profile,'venturi','batch',0,'1,8',1)
  library=candidate_library;b=experiment('screen-shared-'+profile,'venturi','batch',0,'1,8',1)
  gains[profile]={str(x['request_count']):y['aggregate_complete_request_output_tps']/x['aggregate_complete_request_output_tps'] for x,y in zip(a['summary'],b['summary'])}
 report['screen_gains']=gains;save()
 selected_profile='q8-residual';library=baseline_library;experiment('screen-mtp-unshared','venturi','scalar',1,'1',1)
 library=candidate_library;experiment('screen-mtp-shared','venturi','scalar',1,'1',1)
 # Preset screening criteria; no external-speed success inferred from these.
 promote=min(gains['q8-residual'].values())>1.01;report['promoted']=promote;save()
 if promote:
  for profile in ['q8-residual','fp16']:
   selected_profile=profile;counts='1,2,4,8' if profile=='q8-residual' else '1,8'
   library=baseline_library;experiment('repeat-unshared-'+profile,'venturi','batch',0,counts,3)
   library=candidate_library;experiment('repeat-shared-'+profile,'venturi','batch',0,counts,3)
  selected_profile='q8-residual';library=baseline_library;experiment('repeat-mtp-unshared','venturi','scalar',1,'1',3)
  library=candidate_library;experiment('repeat-mtp-shared','venturi','scalar',1,'1',3)
  experiment('repeat-llama-tensor-off','llama','tensor',0,'1,2,4,8',3)
  try:experiment('repeat-llama-tensor-mtp1-c1','llama','tensor',1,'1',3)
  except Exception:report['external_mtp_failure_preserved']=True;save()
 report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
