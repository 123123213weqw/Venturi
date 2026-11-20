import ast,pathlib,subprocess
p=pathlib.Path('/tmp/venturi-model-profile-fixture-dispatch.py');tree=ast.parse(p.read_text());worker=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='worker' for t in n.targets)))
worker=worker.replace('model-profile-src-v1','shared-input-src-v2').replace('venturi-model-profile-fixtures-v1','venturi-shared-input-final-v4');a=worker.index(" assert (root/");b=worker.index(" (out/'completed.json')",a)
body=r'''
 gate=root/'artifacts/venturi-shared-input-campaign-v3';dependency_pid=json.loads((gate/'process.json').read_text())['pid'];deadline=time.monotonic()+900
 while not (gate/'exit-code.txt').exists():
  proc=pathlib.Path(f'/proc/{dependency_pid}/cmdline');assert proc.exists() and str(gate/'worker.py').encode() in proc.read_bytes(),'Pinned predecessor stopped without terminal evidence'
  assert time.monotonic()<deadline,'Bounded predecessor wait elapsed'
  (out/'dependency-wait.json').write_text(json.dumps(dict(pid=dependency_pid,confirmed_live_unix=time.time(),command=str(gate/'worker.py')))+'\n');time.sleep(5)
 assert (gate/'exit-code.txt').read_text().strip()=='0';assert json.loads((gate/'campaign-report.json').read_text())['promoted']
 previous=root/'artifacts/venturi-shared-input-fixtures-v1';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_SHARED_LIBRARY=library,VENTURI_SHARED_BASELINE_LIBRARY=str(root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'),VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-q','--tb=short'],'tests.log',timeout=300)
 case=json.loads((root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json').read_text());case['expected_generated_ids']=case['expected_generated_ids'][:8];(out/'profile-case.json').write_text(json.dumps(case,indent=2)+'\n');target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'
 nsys=subprocess.check_output(['which','nsys'],text=True).strip();traces=[]
 for profile in ['fp16','q8-residual']:
  for count,mode in [(1,'scalar'),(8,'batch')]:
   name=f'{profile}-c{count}';prefix=out/name
   run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_qwen_batch.py',str(target),'--library',library,'--case',str(out/'profile-case.json'),'--mode',mode,'--count',str(count),'--projection-profile',profile,'--output',str(out/(name+'-metadata.json'))],name+'-profile.log',timeout=240)
   run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=120)
   run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=90)
   traces+=['--trace',name+'='+str(prefix)+'.sqlite']
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),*traces,'--output',str(out/'resources.json')],'resources.log',timeout=120)
 retained={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.is_file() and p.suffix in ['.sqlite','.nsys-rep']};(out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
'''
worker=worker[:a]+body+worker[b:];compile(worker,'worker.py','exec')
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-shared-input-final-v4');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -'];r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
