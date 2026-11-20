try:
 import pathlib,subprocess,os,time,json,hashlib,signal,traceback
 root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/strided-src-v2';out=root/'artifacts/venturi-strided-final-v6';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
 def stop(p):
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 def run(cmd,name,timeout=300):
  (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
  with (out/name).open('wb') as log:
   p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=p.wait(timeout=timeout)
   except BaseException:stop(p);raise
   if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
 try:

  gate=root/'artifacts/venturi-strided-campaign-v5';dependency_pid=json.loads((gate/'process.json').read_text())['pid'];deadline=time.monotonic()+900
  while not (gate/'exit-code.txt').exists():
   proc=pathlib.Path(f'/proc/{dependency_pid}/cmdline');assert proc.exists() and str(gate/'worker.py').encode() in proc.read_bytes(),'Pinned predecessor stopped without terminal evidence'
   assert time.monotonic()<deadline,'Bounded predecessor wait elapsed'
   (out/'dependency-wait.json').write_text(json.dumps(dict(pid=dependency_pid,confirmed_live_unix=time.time(),command=str(gate/'worker.py')))+'\n');time.sleep(5)
  assert (gate/'exit-code.txt').read_text().strip()=='0';assert json.loads((gate/'campaign-report.json').read_text())['promoted']
  previous=root/'artifacts/venturi-strided-fixtures-v3';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text());assert hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest()==build['library_sha256']
  for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
  baseline=str(root/'artifacts/venturi-wmma-guarded-v4/build/8537a2e208c36c23/libventuri_blocks.so')
  env.update(VENTURI_WMMA_LIBRARY=library,VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=baseline,VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_SHARED_LIBRARY=library,VENTURI_SHARED_BASELINE_LIBRARY=str(root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'),VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
  run([py,'-m','pytest','-q','--tb=short'],'tests.log',timeout=300)
  target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';nsys=subprocess.check_output(['which','nsys'],text=True).strip()
  case=json.loads((root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json').read_text());case['expected_generated_ids']=case['expected_generated_ids'][:8];(out/'profile-case.json').write_text(json.dumps(case,indent=2)+'\n');traces=[]
  for count,mode in [(1,'scalar'),(8,'batch')]:
   name=f'q8-residual-c{count}';prefix=out/name
   run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_qwen_batch.py',str(target),'--library',library,'--case',str(out/'profile-case.json'),'--mode',mode,'--count',str(count),'--projection-profile','q8-residual','--output',str(out/(name+'-metadata.json'))],name+'-profile.log',timeout=240)
   run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=120)
   run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=90)
   traces+=['--trace',name+'='+str(prefix)+'.sqlite']
  (out/'profile_mtp.py').write_text('"""Separate MTP1 timeline: diagnostics only, not serving throughput."""\nimport argparse, json\nfrom pathlib import Path\nimport torch\nfrom venturi.models import NativeQwenModel, NativeQwenMTPModel, QwenQ8Weights, QwenMTPWeights\nfrom venturi.ninfer import NInferArtifact\nfrom venturi.ops import CudaBackend\np=argparse.ArgumentParser(description=__doc__)\np.add_argument(\'--library\',required=True);p.add_argument(\'--case\',required=True);p.add_argument(\'--output\',required=True);a=p.parse_args()\nr=Path(\'/home/data/wangyue/repos/codex-build/Venturi\');case=json.loads(Path(a.case).read_text());torch.set_num_threads(4)\ntarget=NativeQwenModel.from_q8(QwenQ8Weights.from_ninfer(NInferArtifact(r/\'models/qwen3.5-27b-text-q8.ninfer\')),CudaBackend(a.library),weight_mode=\'resident\',dtype=torch.float16,device=\'cuda:0\',max_tokens=32,tensor_parallel_device=\'cuda:1\',tensor_parallel_transport=\'host\',projection_profile=\'q8-residual\')\nmodel=NativeQwenMTPModel.from_q8(target,QwenMTPWeights.from_ninfer(NInferArtifact(r/\'models/qwen3.5-27b-mtp-q8.ninfer\')))\nsession=model.create_session(256)\ntry:\n expected=case[\'expected_generated_ids\'];prompt=case[\'prompt_token_ids\'];assert session.generate(prompt,max_new_tokens=len(expected))==expected;session.reset()\n with torch.cuda.nvtx.range(\'native_unsplit\'):\n  actual=session.generate(prompt,max_new_tokens=len(expected))\n assert actual==expected\n Path(a.output).write_text(json.dumps(dict(scope=__doc__,profiled=True,logical_requests=1,mtp_depth=1,projection_profile=\'q8-residual\',draft_projection_profile=\'fp16\',output_ids=actual,mtp_stats=session.stats()),indent=2)+\'\\n\')\nfinally:session.close();model.close();target.close()\n')
  (out/'copy_attribution.py').write_text('import pathlib,json,sqlite3,hashlib\nroot=pathlib.Path(\'/home/data/wangyue/artifacts\');base=root/\'venturi-strided-final-v6\'\npaths={**{f\'{label}-c{c}\':root/folder/f\'q8-residual-c{c}.sqlite\' for label,folder in [(\'control\',\'venturi-wmma-final-v6\'),(\'candidate\',\'venturi-strided-final-v6\')] for c in [1,8]},**{label+\'-mtp-c1\':base/(label+\'-mtp-c1.sqlite\') for label in [\'control\',\'candidate\']}}\nresult={};partitions={}\nfor label,p in paths.items():\n db=sqlite3.connect(p.resolve().as_uri()+\'?mode=ro\',uri=True);db.row_factory=sqlite3.Row;strings=dict(db.execute(\'select id,value from StringIds\'));events=[e for e in db.execute(\'select * from NVTX_EVENTS where end is not null\') if (e[\'text\'] or strings.get(e[\'textId\']))==\'native_unsplit\'];tables={r[0] for r in db.execute("select name from sqlite_master where type=\'table\'")};apis=[]\n for table in [\'CUPTI_ACTIVITY_KIND_RUNTIME\',\'CUPTI_ACTIVITY_KIND_DRIVER\']:\n  if table in tables:apis.extend(db.execute(\'select * from \'+table))\n mem=list(db.execute(\'select * from CUPTI_ACTIVITY_KIND_MEMCPY\'));rows=[];groups={}\n for e in events:\n  calls=[a for a in apis if a[\'globalTid\']==e[\'globalTid\'] and e[\'start\']<=a[\'start\'] and a[\'end\']<=e[\'end\']];ids={a[\'correlationId\'] for a in calls};copies=[c for c in mem if c[\'correlationId\'] in ids];kinds={c[\'correlationId\']:c[\'copyKind\'] for c in copies};twoD=[a for a in calls if \'Memcpy2DAsync\' in strings[a[\'nameId\']]]\n  rows.append(dict(range=\'native_unsplit\',host_nanoseconds=e[\'end\']-e[\'start\'],cuda_memcpy2d_api_calls=len(twoD),cuda_memcpy2d_cpu_nanoseconds=sum(a[\'end\']-a[\'start\'] for a in twoD),copies={str(k):dict(count=sum(c[\'copyKind\']==k for c in copies),bytes=sum(c[\'bytes\'] for c in copies if c[\'copyKind\']==k),summed_nanoseconds=sum(c[\'end\']-c[\'start\'] for c in copies if c[\'copyKind\']==k)) for k in [1,2,8]}))\n  for a in calls:\n   name=strings[a[\'nameId\']]\n   if \'Memcpy\' not in name:continue\n   key=f"{name}/kind={kinds.get(a[\'correlationId\'],\'unjoined\')}";d=groups.setdefault(key,dict(count=0,api_nanoseconds=0));d[\'count\']+=1;d[\'api_nanoseconds\']+=a[\'end\']-a[\'start\']\n meta=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope=\'Read-only API/copy-correlation attribution restricted to the submitting-thread native range. CPU API sums can contain dependency waits and nested API time; device copies can overlap. Neither is an elapsed-time savings or achieved-bandwidth measurement.\')\n result[label]=dict(**meta,ranges=rows);partitions[label]=dict(**meta,groups=groups);db.close()\n(base/\'copy-audit.json\').write_text(json.dumps(result,indent=2)+\'\\n\');(base/\'copy-api-partition.json\').write_text(json.dumps(partitions,indent=2)+\'\\n\')\n')
  for label,selected in [('control',baseline),('candidate',library)]:
   name=label+'-mtp-c1';prefix=out/name
   run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,str(out/'profile_mtp.py'),'--library',selected,'--case',str(out/'profile-case.json'),'--output',str(out/(name+'-metadata.json'))],name+'-profile.log',timeout=240)
   run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=120)
   run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=90)
   if label=='candidate':traces+=['--trace',name+'='+str(prefix)+'.sqlite']
  run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),*traces,'--output',str(out/'resources.json')],'resources.log',timeout=120)
  run([py,str(out/'copy_attribution.py')],'copy-attribution.log',timeout=120)
  retained={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.is_file() and p.suffix in ['.sqlite','.nsys-rep']};(out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+'\n')
  (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
  (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
  run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
  (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\n');rc=0
 except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
 (out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
except Exception:
 import pathlib,traceback
 out=pathlib.Path('/home/data/wangyue/artifacts/venturi-strided-final-v6')
 (out/'failure.txt').write_text(traceback.format_exc())
 (out/'exit-code.txt').write_text('1\n')
 raise
