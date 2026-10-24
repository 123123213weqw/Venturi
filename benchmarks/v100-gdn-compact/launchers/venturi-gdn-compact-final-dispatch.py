from pathlib import Path
import subprocess
base=Path('/tmp/venturi-tiled-staging-final-worker.py').read_text().replace('tiled-staging-src-v4','gdn-compact-src-v2').replace('venturi-tiled-staging-final-v5','venturi-gdn-compact-final-v4').replace('venturi-tiled-staging-campaign-v4','venturi-gdn-compact-campaign-v3').replace('venturi-tiled-staging-model-v3','venturi-gdn-compact-model-v2')
base=base.replace("baseline=str(root/'artifacts/venturi-shared-activation-model-v2/build/43feb26c8cde610e/libventuri_blocks.so')", "baseline=str(root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so')")
a=base.index(' env.update(VENTURI_Q8_SHARED_LIBRARY=library,');b=base.index(" target=root/",a)
base=base[:a]+''' (out/'reused-fixture-tests.json').write_text(json.dumps(dict(path=str(previous),sha256=hashlib.sha256((previous/'full-tests.log').read_bytes()).hexdigest(),scope='Same exact library and nine sources: full fixture suite reused, not rerun.'),indent=2)+'\\n')
''' +base[b:]
base=base.replace("[('control',4,'batch',baseline),('candidate',1,'scalar',library),('candidate',2,'batch',library),('candidate',4,'batch',library),('candidate',8,'batch',library)]", "[('candidate',1,'scalar',library),('candidate',8,'batch',library)]")
needle=" for label,count,mode,selected in "
a=base.index(needle)
base=base[:a]+''' old=root/'artifacts/venturi-projection-fusion-final-v5';assert hashlib.sha256((out/'profile-case.json').read_bytes()).hexdigest()==hashlib.sha256((old/'profile-case.json').read_bytes()).hexdigest()
 (out/'reused-control-profiles.json').write_text(json.dumps({name:dict(path=str(old/name),sha256=hashlib.sha256((old/name).read_bytes()).hexdigest(),scope='Explicitly reused unchanged 23-prompt/8-output control on same hardware/library/workload; not a new execution.') for name in ['candidate-c1-attribution.json','candidate-c8-attribution.json','candidate-mtp-c1-attribution.json','resources.json','copy-audit.json','api-audit.json']},indent=2)+'\\n')
''' +base[a:]
Path('/tmp/venturi-gdn-compact-final-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-gdn-compact-final-v4');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
