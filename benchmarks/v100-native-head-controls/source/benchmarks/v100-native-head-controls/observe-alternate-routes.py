import subprocess,json
from pathlib import Path
b=Path('benchmarks/v100-native-head-controls')
r=subprocess.run(['tailscale','ping','--c=3','--timeout=5s','100.70.188.115'],capture_output=True,text=True,timeout=25);(b/'tailscale-target-ping.log').write_text(r.stdout+r.stderr);print('TAIL',r.returncode,r.stdout,r.stderr,flush=True)
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','WZU_Server_Rescue','python3 -']
code="""import pathlib,json,socket
p=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-head-controls-v1');print(json.dumps(dict(host=socket.gethostname(),exists=p.exists(),process=(p/'process.json').read_text() if (p/'process.json').exists() else None)))
"""
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=50);(b/'rescue-route-observation.log').write_text(r.stdout+r.stderr);print('RESCUE',r.returncode,r.stdout,r.stderr,flush=True)
if r.returncode:
 ssh[-2]='WZU_Server_Via_4080';r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);(b/'existing-jump-route-observation.log').write_text(r.stdout+r.stderr);print('SAME_V100_VIA_EXISTING_4080_JUMP',r.returncode,r.stdout,r.stderr,flush=True)
if not r.returncode:
 d=json.loads(r.stdout)
 (b/'healthy-route.json').write_text(json.dumps(dict(ssh=ssh,observation=d),indent=2)+'\n')
