import pathlib,subprocess
old=pathlib.Path('benchmarks/v100-native-batch/launchers/venturi-batch-campaign-worker-v4.py').read_text();prefix=old[:old.index('try:\n assert (prior')];prefix=prefix.replace("out=root/'artifacts/venturi-native-batch-v4'","out=root/'artifacts/venturi-llama-arithmetic-v1'").replace('port=18944','port=18946')
body=r'''
server=None;child_pid=None
try:
 gate=root/'artifacts/venturi-tp-worker-screen-v1'
 for _ in range(180):
  if (gate/'exit-code.txt').exists():break
  time.sleep(5)
 else:raise RuntimeError('Bounded live worker wait elapsed; no competitor profile launched')
 # A terminal failed experiment is also safe to follow; do not infer success.
 report.update(scope='Instrumented arithmetic-path audit, not throughput. Whole-process CUDA graph node trace includes startup and 1/8-request warmup/timed calls, 23 input IDs and 8 outputs. Only kernel identities/loaded-source hints are claimed.',preceding_worker_exit_code=(gate/'exit-code.txt').read_text().strip(),profiled=True,llama_binary_sha256=digest(bindir/'llama-server'),llama_build_info=(bindir.parent/'BUILD_INFO').read_text());save()
 nsys=subprocess.check_output(['which','nsys'],text=True).strip();prefix_path=out/'trace'
 server_cmd=[str(bindir/'llama-server'),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-ngl','all','--split-mode','tensor','--tensor-split','1,1','--fit','off','-c','2048','--parallel','8','-b','256','-ub','256','-t','4','-tb','4','-fa','on','-ctk','f16','-ctv','f16','--backend-sampling','--spec-type','none']
 cmd=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--cuda-graph-trace=node','--force-overwrite=false','--output',str(prefix_path),*server_cmd];(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/'profile.log').open('wb') as f:server=subprocess.Popen(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 def descendants(pid):
  pending=[pid];found=[]
  while pending:
   current=pending.pop();path=pathlib.Path(f'/proc/{current}/task/{current}/children')
   if path.exists():
    children=[int(p) for p in path.read_text().split()];pending+=children;found+=children
  return found
 for _ in range(180):
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
 run([py,'tools/benchmark_completion_api.py','--url',f'http://127.0.0.1:{port}','--server-pid',str(child_pid),'--case',str(out/'case.json'),'--concurrency','1,8','--repeats','1','--output-directory',str(out/'instrumented-http'),'--expect-pinned-ids'],out/'instrumented-client.log',timeout=120)
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
'''
worker=prefix+body
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-llama-arithmetic-v1');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -'];r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
