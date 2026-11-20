from pathlib import Path
import subprocess
root=Path('benchmarks/v100-activation-words/v100');root.mkdir(exist_ok=True)
ssh='ssh -o BatchMode=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o ProxyCommand=none -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
for name in ['study-v1','micro-profile-v2','bound-study-v3','bound-profile-v4','aos-five-study-v5','aos-five-profile-v6','guarded-study-v7','guarded-model-v8','guarded-campaign-v9','guarded-final-v10']:
 p=root/name;p.mkdir(exist_ok=True)
 r=subprocess.run(['rsync','-az','--exclude=*-sass.txt','--exclude=*.sqlite','--exclude=*.nsys-rep','--exclude=*.so','--exclude=*.ninfer','--exclude=*.gguf','--exclude=*.cubin','--exclude=__pycache__/','-e',ssh,'WZU_Server:/home/data/wangyue/artifacts/venturi-activation-words-'+name+'/',str(p)+'/'],capture_output=True,text=True,timeout=90)
 if r.returncode:print(name,r.returncode,r.stderr);r.check_returncode()
 print(name,'fetched',flush=True)
