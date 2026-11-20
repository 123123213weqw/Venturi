from pathlib import Path
import subprocess,json,time
b=Path('benchmarks/v100-native-head-controls');n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);ssh=n['SSH'][:];ssh[ssh.index('ControlPath=none')]='ControlPath=/Users/wangyue/.ssh/cm-14974355c3f315e78f58843cae0595e186514110';ssh[ssh.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6'
code="""import pathlib,json,socket
base=pathlib.Path('/home/data/wangyue/artifacts');r={'host':socket.gethostname()}
for name in ['venturi-native-head-controls-v4','venturi-native-head-aligned-v5']:
 p=base/name;d={'exists':p.exists()}
 for f in ['process.json','exit-code.txt','failure.txt']:
  if (p/f).exists():d[f]=(p/f).read_text()[-2500:]
 if (p/'process.json').exists():
  q=pathlib.Path('/proc')/str(json.loads((p/'process.json').read_text())['pid'])/'cmdline';d['live_cmdline']=q.read_bytes().decode(errors='replace') if q.exists() else None
 r[name]=d
print(json.dumps(r))
"""
try:r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=70)
except subprocess.TimeoutExpired as e:(b/'unknown-v4-v5-outcome-observation.log').write_text(str(e));print('Observation timed out; outcomes remain unknown, no replay',flush=True);raise SystemExit(0)
(b/'unknown-v4-v5-outcome-observation.log').write_text(r.stdout+r.stderr);print(r.returncode,r.stdout,r.stderr,flush=True)
if not r.returncode:(b/'v4-v5-authoritative-state.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
