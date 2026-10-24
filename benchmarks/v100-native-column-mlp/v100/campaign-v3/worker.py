import pathlib, subprocess, json, os, time, traceback, signal, urllib.request, hashlib, datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-native-column-mlp-campaign-v3';src=root/'repos/codex-build/Venturi/native-column-mlp-src-v2';prior=root/'artifacts/venturi-native-column-mlp-model-v2'
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
def experiment(label,engine,mode,depth,counts,repeats,recovery="replay"):
 directory=out/label;directory.mkdir();server=None;samplers=[];logs=[];item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,mtp_recovery=recovery,projection_rows=selected_rows,projection_profile=selected_profile if engine=='venturi' else None,library_sha256=digest(library) if engine=='venturi' else None,candidate_build=(library==candidate_library) if engine=='venturi' else None,completed=False)
 try:
  for gpu in [1,0]:
   f=(directory/f'gpu-{gpu}-samples.csv').open('wb');logs.append(f);samplers.append(subprocess.Popen(['nvidia-smi','-i',str(gpu),'--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits','-lms','100'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
  if engine=='llama':
   cmd=[str(bindir/'llama-server'),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-ngl','all','--split-mode',mode,'--tensor-split','1,1','--fit','off','-c','2048','--parallel','8','-b','256','-ub','256','-t','4','-tb','4','-fa','on','-ctk','f16','-ctv','f16','--backend-sampling','--spec-type','draft-mtp' if depth else 'none']
   if depth:cmd+=['--spec-draft-n-max',str(depth),'--spec-draft-backend-sampling']
  else:
   cmd=[py,'tools/serve_native_benchmark.py',str(target),'--library',str(library),'--mode','tp','--port',str(port),'--slots','8','--capacity','256','--ready-file',str(directory/'ready.json')]
   cmd+=['--projection-profile',selected_profile,'--projection-rows',str(selected_rows)]
   if depth:cmd+=['--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--mtp-depth',str(depth),'--mtp-recovery',recovery]
   if mode=='batch':cmd+=['--batch-target','--batch-window-ms','25']
   if mode=='mtp-batch':cmd+=['--batch-mtp','--batch-window-ms','25']
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
  if engine=='venturi' and mode in ['batch','mtp-batch']:
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



candidate_library=library
legacy=root/'artifacts/venturi-native-mtp2-prefix-v5/build/48a678383e862400/libventuri_blocks.so'
cohort_control=root/'artifacts/venturi-batched-mtp-model-v2/build/e94e01699eaad5b0/libventuri_blocks.so'
selected_profile='q8-residual';selected_rows=32
try:
 assert digest(legacy)=='9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894'
 assert digest(cohort_control)=='f07bf75a975b963911aff5fefd5614dbf0b8b5b74e499f5525b6ad5d5317369a'
 actual={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert actual==json.loads((prior/'pre-execution-source.json').read_text())
 model=json.loads((prior/'completed.json').read_text());assert model['model_qualified']
 q=json.loads((prior/'qualification/report.json').read_text());assert q['completed'] and q['profiles']['q8-residual-column-mlp']['candidate']['passed']
 q=json.loads((prior/'mtp-batch-qualification/report.json').read_text());assert q['completed'] and q['selective_reset_untouched_state_exact']
 q=json.loads((prior/'mtp2-qualification/report.json').read_text());assert q['completed'] and len(q['cases'])==4 and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['committed_draft_state_equal_mtp1'] for c in q['cases'])
 build=json.loads(candidate_library.with_name('build.json').read_text());assert len(build['source_sha256'])==10
 for rel,h in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==h
 report.update(scope='Native MLP-only column/row TP: bit-preserving down columns, FP32 partial/host transport and sum then FP16; changed association separately teacher/state-qualified. Original FP16 draft and row-TP attention/GDN unchanged. Global HTTP cohort includes reset/admission/queue/reply/client archival; native timing excludes those. No microbenchmark-to-TPS inference.',model_qualified=True,source_sha256=actual,candidate_library_sha256=digest(candidate_library),path_promotions={},screen_ratios={},projection_profiles=dict(control='q8-residual',candidate='q8-residual-column-mlp'))
 paths=[('mtp2','scalar',2,'prefix',32,legacy),('ordinary256','batch',0,'replay',256,cohort_control),('mtp-batch','mtp-batch',1,'replay',32,cohort_control)]
 for name,mode,depth,recovery,rows,control_library in paths:
  selected_rows=rows;selected_profile='q8-residual';library=control_library
  control=experiment('screen-control-'+name,'venturi',mode,depth,'1,8',1,recovery)
  selected_profile='q8-residual-column-mlp';library=candidate_library
  candidate=experiment('screen-candidate-'+name,'venturi',mode,depth,'1,8',1,recovery)
  av={v['request_count']:v for v in control['summary']};bv={v['request_count']:v for v in candidate['summary']};metric='aggregate_complete_request_output_tps'
  ratios={str(n):bv[n][metric]/av[n][metric] for n in bv}
  report['screen_ratios'][name]=ratios;report['path_promotions'][name]=ratios['1']>1.01 or ratios['8']>1.02;save()
 # All screen arms finish before conditional matrices, so weak paths are visible.
 for name,mode,depth,recovery,rows,control_library in paths:
  if not report['path_promotions'][name]:continue
  selected_rows=rows;selected_profile='q8-residual';library=control_library
  experiment('repeat-control-'+name,'venturi',mode,depth,'1,2,4,8',3,recovery)
  selected_profile='q8-residual-column-mlp';library=candidate_library
  experiment('repeat-candidate-'+name,'venturi',mode,depth,'1,2,4,8',3,recovery)
 if any(report['path_promotions'].values()):
  experiment('repeat-llama-mtp2-fresh','llama','tensor',2,'1',3,'prefix')
  experiment('repeat-llama-nonspec-fresh','llama','tensor',0,'1,2,4,8',3)
 report['reused_controls']=[]
 for p in [root/'artifacts/venturi-batched-mtp-campaign-v3/campaign-report.json',root/'artifacts/venturi-batched-mtp-sdk-campaign-v8/campaign-report.json',root/'artifacts/venturi-wmma-campaign-v5/campaign-report.json',root/'artifacts/venturi-strided-external-depth-v9/campaign-report.json']:
  assert json.loads(p.read_text())['completed'];report['reused_controls'].append(dict(path=str(p),sha256=digest(p)))
 report['promotion_rule']='C1 HTTP >1% or C8 >2% admits full 3-repeat 1/2/4/8 path matrix; subsequent adoption also requires no repeated cell >2% regression and two improving qualifying trials. Full external completion is still >=5% on single post-first plus every matched cohort count.'
 report['external_goal_complete']=False;report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
