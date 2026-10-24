import subprocess,json,shlex
from pathlib import Path
b=Path('benchmarks/v100-native-head-controls')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
proxy=shlex.join(['ssh']+opts+['-W','%h:%p','School_4090D'])
ssh=['ssh']+opts+['-o','ProxyCommand='+proxy,'V100_LAN_J','python3 -']
code="""import pathlib,json,socket
p=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-head-controls-v1');print(json.dumps(dict(host=socket.gethostname(),exists=p.exists(),process=(p/'process.json').read_text() if (p/'process.json').exists() else None)))
"""
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);(b/'campus-jump-observation.log').write_text(r.stdout+r.stderr);print(r.returncode,r.stdout,r.stderr,flush=True)
if not r.returncode:(b/'healthy-route.json').write_text(json.dumps(dict(ssh=ssh,observation=json.loads(r.stdout)),indent=2)+'\n')
