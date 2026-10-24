from pathlib import Path
import subprocess
base=Path('/tmp/venturi-tiled-staging-final-worker.py').read_text()
base=base.replace('tiled-staging-src-v4','projection-fusion-src-v3').replace('venturi-tiled-staging-final-v5','venturi-projection-fusion-final-v5').replace('venturi-tiled-staging-campaign-v4','venturi-projection-fusion-campaign-v4').replace('venturi-tiled-staging-model-v3','venturi-projection-fusion-model-v3')
base=base.replace("baseline=str(root/'artifacts/venturi-shared-activation-model-v2/build/43feb26c8cde610e/libventuri_blocks.so')", "baseline=str(root/'artifacts/venturi-tiled-staging-model-v3/build/398e4f4a1dd037ea/libventuri_blocks.so');grouped=str(root/'artifacts/venturi-projection-fusion-v1/build/b08a62aa2ac57cc2/libventuri_blocks.so')")
a=base.index(' env.update(VENTURI_Q8_SHARED_LIBRARY=library,');b=base.index(" target=root/",a)
base=base[:a]+''' (out/'reused-fixture-tests.json').write_text(json.dumps(dict(path=str(previous),sha256=hashlib.sha256((previous/'full-tests.log').read_bytes()).hexdigest(),scope='Same exact candidate native library and nine source files; full fixture suite reused, not rerun.'),indent=2)+'\\n')
''' +base[b:]
base=base.replace("[('control',4,'batch',baseline),('candidate',1,'scalar',library),('candidate',2,'batch',library),('candidate',4,'batch',library),('candidate',8,'batch',library)]", "[('control',1,'scalar',baseline),('control',8,'batch',baseline),('grouped',1,'scalar',grouped),('grouped',8,'batch',grouped),('candidate',1,'scalar',library),('candidate',8,'batch',library)]")
start=base.index(" (out/'remote-retained-files.json')")
base=base[:start]+''' prefix=out/'micro'
 run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/benchmark_q8_fused.py',str(target),'--library',library,'--baseline-library',baseline,'--output-directory',str(out/'micro-results'),'--profile'],'micro-profile.log',timeout=240)
 run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],'micro-export.log',timeout=120)
 run([py,'tools/summarize_q8_projection_trace.py',str(prefix)+'.sqlite','--output',str(out/'micro-attribution.json')],'micro-attribution.log',timeout=90)
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),'--trace','micro='+str(prefix)+'.sqlite','--output',str(out/'micro-resources.json')],'micro-resources.log',timeout=180)
''' +base[start:]
Path('/tmp/venturi-projection-fusion-final-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-projection-fusion-final-v5');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
