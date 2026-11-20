import pathlib,subprocess,json,os,time,traceback,signal,urllib.request,hashlib,datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-llama-comparison-v1';src=root/'repos/codex-build/Venturi/llama-compare-src-v1'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');library=root/'repos/codex-build/Venturi/v100-mtp-build/fb11e6eff12b2fbf/libventuri_blocks.so';target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json'
bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');port=18943
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',NCCL_P2P_DISABLE='1',LD_LIBRARY_PATH=str(bindir)+':/usr/local/cuda/lib64:'+str(root/'repos/codex-build/.venvs/b2-v100/lib/python3.10/site-packages/nvidia/nccl/lib'))
report=dict(completed=False,scope='Initial unprofiled loopback HTTP screen, same original checkpoint and fixed work. One repeat is screening only, not promotion.',experiments=[],physical_gpus=[1,0],cuda_visible_devices='1,0',utc_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds()),achieved_hardware_counters=None)
def save():(out/'screen-report.json').write_text(json.dumps(report,indent=2)+'\n')
def stop(p):
 if p and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run(cmd,path,timeout=600):
 with path.open('w') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:stop(p);raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {path}')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 save();terminal=out/'export-exit-code.txt'
 for _ in range(180):
  if terminal.exists():break
  time.sleep(10)
 else:raise RuntimeError('Bounded export wait elapsed; no GPU process started')
 assert terminal.read_text().strip()=='0','Original-source conversion failed; no GPU process started'
 exported=json.loads((out/'export-report.json').read_text());assert exported['completed'];gguf=exported['model']
 report.update(gguf_sha256=exported['gguf_sha256'],venturi_library_sha256=digest(library),llama_binary_directory=str(bindir),llama_library_hashes={p.name:digest(p) for p in bindir.iterdir() if p.is_file() and not p.is_symlink()},llama_build_info=(bindir.parent/'BUILD_INFO').read_text(),llama_engine_scope='Existing pinned llama.cpp production fork, not latest stock upstream; retain all loaded library hashes.',explicit_nccl_p2p_disable='1',source_hashes={str(p.relative_to(src)):digest(p) for f in ['src','tests','tools'] for p in (src/f).rglob('*.py') if '__pycache__' not in p.parts})
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-before.txt')
 run([py,'-m','pytest','-q','tests/test_completion_benchmark.py','--tb=short'],out/'adapter-tests.log',timeout=120)
 configurations=[('llama-tensor-off','llama','tensor',0),('llama-tensor-mtp1','llama','tensor',1),('llama-tensor-mtp3','llama','tensor',3),('llama-layer-off','llama','layer',0),('venturi-tp-mtp1','venturi','tp',1),('venturi-pp-off','venturi','pp',0)]
 for label,engine,mode,depth in configurations:
  directory=out/label;directory.mkdir();server=None;samplers=[];logs=[];item=dict(label=label,engine=engine,mode=mode,mtp_depth=depth,completed=False)
  try:
   for gpu in [1,0]:
    f=(directory/f'gpu-{gpu}-samples.csv').open('wb');logs.append(f)
    samplers.append(subprocess.Popen(['nvidia-smi','-i',str(gpu),'--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits','-lms','100'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
   if engine=='llama':
    cmd=[str(bindir/'llama-server'),'-m',gguf,'--host','127.0.0.1','--port',str(port),'-ngl','all','--split-mode',mode,'--tensor-split','1,1','--fit','off','-c','2048','--parallel','8','-b','256','-ub','256','-t','4','-tb','4','-fa','on','-ctk','f16','-ctv','f16','--backend-sampling','--spec-type','draft-mtp' if depth else 'none']
    if depth:cmd+=['--spec-draft-n-max',str(depth),'--spec-draft-backend-sampling']
   else:
    cmd=[py,'tools/serve_native_benchmark.py',str(target),'--library',str(library),'--mode',mode,'--port',str(port),'--slots','8','--capacity','256','--ready-file',str(directory/'ready.json')]
    if depth:cmd+=['--mtp-artifact',str(draft)]
   item['command']=cmd;(directory/'command.json').write_text(json.dumps(cmd,indent=2))
   f=(directory/'server.log').open('wb');logs.append(f);server=subprocess.Popen(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);item['pid']=server.pid
   for attempt in range(240):
    if server.poll() is not None:raise RuntimeError(f'Server exited during startup: {server.returncode}')
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
      item['health']=json.load(response);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('Bounded server health wait elapsed')
   (directory/'loaded-maps.txt').write_text('\n'.join(s for s in pathlib.Path(f'/proc/{server.pid}/maps').read_text().splitlines() if '.so' in s))
   cmd=[py,'-u','tools/benchmark_completion_api.py','--url',f'http://127.0.0.1:{port}','--server-pid',str(server.pid),'--case',str(case),'--concurrency','1,8','--repeats','1','--output-directory',str(directory/'screen')]
   if engine=='venturi':cmd+=['--expect-pinned-ids']
   run(cmd,directory/'benchmark.log')
   measured=json.loads((directory/'screen/report.json').read_text());assert measured['completed'];item.update(completed=True,summary=measured['summary'],stable_outputs=measured['all_outputs_equal_first_wave'])
  except BaseException:item['failure']=traceback.format_exc();(directory/'failure.txt').write_text(item['failure'])
  finally:
   stop(server)
   for sampler in samplers:stop(sampler)
   for f in logs:f.close()
   report['experiments'].append(item);save()
  print(json.dumps(item),flush=True)
 report['completed']=all(r['completed'] for r in report['experiments']);save()
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 rc=0 if report['completed'] else 1
except BaseException:(out/'screen-failure.txt').write_text(traceback.format_exc());rc=1
(out/'screen-exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
