from pathlib import Path
import subprocess
base=Path('/tmp/venturi-gdn-compact-final5-worker.py').read_text().replace('gdn-compact-src-v2','prefill-budget-src-v1').replace('venturi-gdn-compact-final-v5','venturi-prefill-budget-final-v3').replace('venturi-gdn-compact-campaign-v3','venturi-prefill-budget-campaign-v2').replace("previous=root/'artifacts/venturi-gdn-compact-model-v2'", "previous=root/'artifacts/venturi-prefill-budget-model-v1'")
base=base.replace("old=root/'artifacts/venturi-projection-fusion-final-v5'", "old=root/'artifacts/venturi-gdn-compact-final-v5'")
base=base.replace("['candidate-c1-attribution.json','candidate-c8-attribution.json','candidate-mtp-c1-attribution.json','resources.json','copy-audit.json']", "['candidate-c1-attribution.json','candidate-c8-attribution.json','candidate-mtp-c1-attribution.json','resources.json']")
base=base.replace("['--projection-profile','q8-residual','--output'", "['--projection-profile','q8-residual','--output'")
# Add the chosen budget to fresh candidate ordinary traces, never to unchanged MTP.
base=base.replace("'--projection-profile','q8-residual','--output',str(out/(name+'-metadata.json'))", "'--projection-profile','q8-residual','--projection-rows',str(chosen),'--output',str(out/(name+'-metadata.json'))")
needle=" previous=root/'artifacts/venturi-prefill-budget-model-v1'"
base=base.replace(needle," chosen=campaign.get('promoted_projection_rows',256)\n"+needle)
a=base.index(" (out/'profile_mtp.py').write_text(");b=base.index(" run([py,'tools/inspect_cuda_occupancy.py'",a)
base=base[:a]+base[b:]
Path('/tmp/venturi-prefill-budget-final-worker.py').write_text(base);compile(base,'final-worker','exec')
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-prefill-budget-final-v3');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
