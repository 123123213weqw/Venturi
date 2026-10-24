import pathlib, subprocess, json, os, time, traceback, signal, urllib.request, hashlib, datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-llama-arithmetic-v2';src=root/'repos/codex-build/Venturi/native-batch-src-v3';prior=root/'artifacts/venturi-native-batch-v2'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');library=pathlib.Path((prior/'library.txt').read_text().strip());target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf';port=18946
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

server=None;child_pid=None
try:
 gate=root/'artifacts/venturi-dp4a-vector-v3'
 for _ in range(180):
  if (gate/'exit-code.txt').exists():break
  time.sleep(5)
 else:raise RuntimeError('Bounded live worker wait elapsed; no competitor profile launched')
 # A terminal failed experiment is also safe to follow; do not infer success.
 report.update(scope='Instrumented arithmetic-path audit, not throughput. Whole-process CUDA graph node trace includes startup and single-request warmup/timed calls, 23 input IDs and 8 outputs. Only kernel identities/loaded-source hints are claimed.',preceding_worker_exit_code=(gate/'exit-code.txt').read_text().strip(),profiled=True,llama_binary_sha256=digest(bindir/'llama-server'),llama_build_info=(bindir.parent/'BUILD_INFO').read_text());save()
 nsys=subprocess.check_output(['which','nsys'],text=True).strip();prefix_path=out/'trace'
 server_cmd=[str(bindir/'llama-server'),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-ngl','all','--split-mode','tensor','--tensor-split','1,1','--fit','off','-c','2048','--parallel','8','-b','256','-ub','256','-t','4','-tb','4','-fa','on','-ctk','f16','-ctv','f16','--backend-sampling','--spec-type','none']
 cmd=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--cuda-graph-trace=node','--force-overwrite=false','--output',str(prefix_path),*server_cmd];(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/'profile.log').open('wb') as f:server=subprocess.Popen(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 def descendants(pid):
  pending=[pid];found=[]
  while pending:
   current=pending.pop();task=pathlib.Path(f'/proc/{current}/task')
   if task.exists():
    children=set()
    for path in task.glob('*/children'):children.update(int(p) for p in path.read_text().split())
    pending+=list(children);found+=list(children)
  return found
 for _ in range(60):
  if server.poll() is not None:raise RuntimeError(f'Owned profiler exited during startup {server.returncode}')
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as f:json.load(f)
   matches=[]
   for pid in descendants(server.pid):
    try:
     cmdline=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');exe=pathlib.Path(f'/proc/{pid}/exe').resolve()
     if exe==bindir/'llama-server' and str(port).encode() in cmdline:matches.append(pid)
    except OSError:pass
   assert len(matches)==1,matches;child_pid=matches[0];break
  except Exception:time.sleep(1)
 else:raise RuntimeError('Bounded profiler/server health wait elapsed')
 report['server_pid']=child_pid;report['profiler_pid']=server.pid;save()
 (out/'loaded-maps.txt').write_text('\n'.join(s for s in pathlib.Path(f'/proc/{child_pid}/maps').read_text().splitlines() if '.so' in s))
 c=json.loads(case.read_text());c['expected_generated_ids']=c['expected_generated_ids'][:8];(out/'case.json').write_text(json.dumps(c,indent=2)+'\n')
 run([py,'tools/benchmark_completion_api.py','--url',f'http://127.0.0.1:{port}','--server-pid',str(child_pid),'--case',str(out/'case.json'),'--concurrency','1','--repeats','1','--output-directory',str(out/'instrumented-http'),'--expect-pinned-ids'],out/'instrumented-client.log',timeout=120)
 assert child_pid in descendants(server.pid),'Owned child identity changed'
 os.kill(child_pid,signal.SIGINT)
 try:rc=server.wait(timeout=180)
 except BaseException:stop(server);raise
 if rc:raise RuntimeError(f'Owned profiler exited {rc}')
 run([nsys,'export','--type','sqlite','--output',str(prefix_path)+'.sqlite',str(prefix_path)+'.nsys-rep'],out/'export.log',timeout=120)
 import sqlite3
 db=sqlite3.connect('file:'+str(prefix_path)+'.sqlite?mode=ro',uri=True);strings=dict(db.execute('select id,value from StringIds'));rows=list(db.execute('select demangledName,count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL group by demangledName'));db.close()
 kernels=[dict(name=strings[i],count=n,instrumented_nanoseconds=duration) for i,n,duration in rows];kernels.sort(key=lambda x:-x['instrumented_nanoseconds']);(out/'kernel-identities.json').write_text(json.dumps(dict(scope='Whole trace including startup/warmup, not per-request attribution or throughput',kernels=kernels),indent=2)+'\n')
 source=root/'repos/codex-build/llama.cpp-meta-tp-candidate-20260829/ggml/src/ggml-cuda';hints={}
 for name in ['mmvq.cu','vecdotq.cuh','quantize.cu','mmq.cu']:
  p=source/name;hints[str(p)]=dict(sha256=digest(p),scope='Source hint from existing release candidate tree, not a recovered exact binary source revision')
 report['source_hints']=hints;report['actual_activation_quantization_kernels']=[k for k in kernels if 'quantize' in k['name'] and ('q8_1' in k['name'] or 'q8_0' in k['name'])];report['actual_quantized_projection_kernels']=[k for k in kernels if 'mul_mat' in k['name'] and ('q8' in k['name'] or 'q<(' in k['name'])]
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.suffix in ('.sqlite','.nsys-rep')},indent=2)+'\n')
 report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 stop(server)
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
