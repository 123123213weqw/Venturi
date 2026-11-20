from pathlib import Path
import subprocess,shlex,json,time
b=Path('benchmarks/v100-native-head-controls');ssh=json.loads((b/'turn3-healthy-route.json').read_text())['ssh']
for tag,remote in [('controls-v1','venturi-native-head-controls-v1'),('aligned-v2','venturi-native-head-aligned-v2')]:
 dest=b/'v100'/tag;dest.mkdir(parents=True,exist_ok=True)
 cmd=['rsync','-az','--partial','--timeout=60','-e',shlex.join(ssh[:-2])]
 for p in ['*.so','*.pt','*.sqlite','*.nsys-rep','core','core.*','__pycache__/']:cmd+=['--exclude',p]
 cmd+=['WZU_Server:/home/data/wangyue/artifacts/'+remote+'/',str(dest)+'/']
 for attempt in range(2):
  if attempt:time.sleep(5)
  try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=210)
  except subprocess.TimeoutExpired as e:
   with (b/'fetch-initial.log').open('a') as f:f.write(tag+' '+str(attempt)+' '+str(e)+'\n')
   print(tag,attempt,'bounded transport timeout; preserve partial transfer',flush=True);continue
  with (b/'fetch-initial.log').open('a') as f:f.write(tag+' '+str(attempt)+' rc'+str(r.returncode)+'\n'+r.stdout+r.stderr)
  print(tag,attempt,r.returncode,r.stdout,r.stderr,flush=True)
  if not r.returncode:break
