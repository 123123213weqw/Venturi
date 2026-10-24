from pathlib import Path
import subprocess,json,time
b=Path('benchmarks/v100-native-head-controls');n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);base=n['SSH'][:];base[base.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6'
code="""import pathlib,json,socket
base=pathlib.Path('/home/data/wangyue/artifacts');r={'host':socket.gethostname()}
for name in ['venturi-native-head-controls-v4','venturi-native-head-aligned-v5']:
 p=base/name;d={'exists':p.exists()}
 for f in ['process.json','exit-code.txt','failure.txt']:
  if (p/f).exists():d[f]=(p/f).read_text()[-2500:]
 if (p/'process.json').exists():
  q=pathlib.Path('/proc')/str(json.loads((p/'process.json').read_text())['pid'])/'cmdline';d['live_cmdline']=q.read_bytes().decode(errors='replace') if q.exists() else None
 if (p/'campaign-report.json').exists():
  c=json.loads((p/'campaign-report.json').read_text());d['campaign']={'completed':c['completed'],'experiments':[(x['label'],x['completed']) for x in c['experiments']]}
 r[name]=d
print(json.dumps(r))
"""
for route in ['direct','master']:
 ssh=base[:]
 if route=='master':ssh[ssh.index('ControlPath=none')]='ControlPath=/Users/wangyue/.ssh/cm-14974355c3f315e78f58843cae0595e186514110'
 try:r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=45)
 except subprocess.TimeoutExpired as e:(b/f'turn3-{route}-observation.log').write_text(str(e));print(route,'observation timeout, no replay',flush=True);continue
 (b/f'turn3-{route}-observation.log').write_text(r.stdout+r.stderr);print(route,r.returncode,r.stdout,r.stderr,flush=True)
 if not r.returncode:
  d=json.loads(r.stdout);(b/'turn3-authoritative-state.json').write_text(json.dumps(d,indent=2)+'\n');(b/'turn3-healthy-route.json').write_text(json.dumps({'ssh':ssh,'observed_unix':time.time()},indent=2)+'\n');break
