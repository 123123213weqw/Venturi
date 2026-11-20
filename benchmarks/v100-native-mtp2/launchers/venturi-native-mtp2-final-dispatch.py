from pathlib import Path
import subprocess
base=Path('/tmp/venturi-gdn-compact-final5-worker.py').read_text().replace('gdn-compact-src-v2','native-mtp2-src-v3').replace('venturi-gdn-compact-final-v5','venturi-native-mtp2-final-v4').replace('venturi-gdn-compact-campaign-v3','venturi-native-mtp2-campaign-v3').replace("previous=root/'artifacts/venturi-gdn-compact-model-v2'", "previous=root/'artifacts/venturi-native-mtp2-model-v2'")
a=base.index(" old=root/");b=base.index(" (out/'remote-retained-files.json')",a)
base=base[:a]+''' for depth in [1,2]:
  for count in [1,8]:
   name=f'mtp{depth}-c{count}';prefix=out/name
   run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(prefix),py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth',str(depth),'--count',str(count),'--output',str(out/(name+'-metadata.json'))],name+'-profile.log',timeout=300)
   run([nsys,'export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],name+'-export.log',timeout=120)
   run([py,'tools/summarize_native_parallel_trace.py',str(prefix)+'.sqlite','--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=90)
   traces+=['--trace',name+'='+str(prefix)+'.sqlite']
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),*traces,'--output',str(out/'resources.json')],'resources.log',timeout=180)
''' +base[b:]
Path('/tmp/venturi-native-mtp2-final-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-mtp2-final-v4');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
