from pathlib import Path
import subprocess
base=Path('benchmarks/v100-token16-reuse/v100');base.mkdir(exist_ok=True)
ssh='ssh -o BatchMode=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o ProxyCommand=none -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
for name in ['study-v1','model-v2','campaign-v3','final-v4','staged-study-v5','staged-model-v6','staged-campaign-v7','staged-final-v8','graph-probe-v9']:
 p=base/name;p.mkdir(exist_ok=True)
 r=subprocess.run(['rsync','-az','--include=*-target-sass.txt','--exclude=*-sass.txt','--exclude=*.sqlite','--exclude=*.nsys-rep','--exclude=*.so','--exclude=*.ninfer','--exclude=*.gguf','--exclude=*.cubin','--exclude=__pycache__/','-e',ssh,'WZU_Server:/home/data/wangyue/artifacts/venturi-token16-'+name+'/',str(p)+'/'],capture_output=True,text=True,timeout=100)
 if r.returncode:print(r.stderr);r.check_returncode()
 print(name,'fetched',flush=True)
