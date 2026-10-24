from pathlib import Path
import subprocess,json
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
ssh=n['SSH'][:];ssh[ssh.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6'
code="""import pathlib,json,time
base=pathlib.Path('/home/data/wangyue/artifacts');names=['venturi-native-head-controls-v1','venturi-native-head-aligned-v2'];previous=None;deadline=time.monotonic()+1800
while True:
 state={};done=True
 for name in names:
  o=base/name;d={}
  if (o/'exit-code.txt').exists():d['exit_code']=int((o/'exit-code.txt').read_text());d['failure']=(o/'failure.txt').read_text()[-2000:] if (o/'failure.txt').exists() else None
  else:
   done=False;pid=json.loads((o/'process.json').read_text())['pid'];p=pathlib.Path(f'/proc/{pid}/cmdline');d['live']=p.exists() and str(o/'worker.py').encode() in p.read_bytes();assert d['live'],'Missing worker handle without terminal marker'
  if (o/'campaign-report.json').exists():
   report=json.loads((o/'campaign-report.json').read_text());d['campaign']=[dict(label=x['label'],complete=x['completed'],summary=x.get('summary')) for x in report.get('experiments',[])];d['promotions']=report['path_promotions']
  if (o/'qualification256/report.json').exists():d['qualified256']=json.loads((o/'qualification256/report.json').read_text())['completed']
  if (o/'completed.json').exists():d['completed']=json.loads((o/'completed.json').read_text())
  state[name]=d
 text=json.dumps(state)
 if text!=previous:print(text,flush=True);previous=text
 if done:break
 assert time.monotonic()<deadline,'Observation timeout; jobs are not stopped or restarted'
 time.sleep(15)
"""
with Path('benchmarks/v100-native-head-controls/live-monitor.log').open('a') as log:
 p=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);p.stdin.write(code);p.stdin.close()
 for line in p.stdout:log.write(line);log.flush();print(line,flush=True,end='')
 rc=p.wait();print('Monitor terminal',rc,flush=True)
 if rc:raise SystemExit(rc)
