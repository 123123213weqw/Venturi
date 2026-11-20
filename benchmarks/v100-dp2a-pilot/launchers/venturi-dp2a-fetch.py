from pathlib import Path
import json,subprocess,shlex,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
out=Path.cwd()/'benchmarks/v100-dp2a-pilot';(out/'v100').mkdir(exist_ok=True);ssh=ns['SSH'][:-2]
for label in ['fixtures-v1','campaign-v2']:
 dest=out/'v100'/label;dest.mkdir(exist_ok=True)
 cmd=['rsync','-az','-e',shlex.join(ssh)]
 for pattern in ['*.so','*.sqlite','*.nsys-rep','*.pt','pilot-sass.txt','__pycache__/']:cmd+=['--exclude',pattern]
 cmd+=['WZU_Server:/home/data/wangyue/artifacts/venturi-dp2a-pilot-'+label+'/',str(dest)+'/']
 for attempt in range(2):
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
  if not p.returncode:break
  if attempt==1:p.check_returncode()
  time.sleep(3)
 print(label,(dest/'exit-code.txt').read_text().strip())
