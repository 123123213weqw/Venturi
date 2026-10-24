from pathlib import Path
import subprocess,json
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
code="""import pathlib,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-head-controls-v1');d={}
for f in ['process.json','exit-code.txt','failure.txt']:
 if (out/f).exists():d[f]=(out/f).read_text()
if 'process.json' in d:
 pid=json.loads(d['process.json'])['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');d['live_cmdline']=q.read_bytes().decode(errors='replace') if q.exists() else None
for sub in ['qualification256/report.json','campaign-report.json']:
 if (out/sub).exists():
  r=json.loads((out/sub).read_text());d[sub]={'completed':r['completed'],'experiments':[{k:v for k,v in x.items() if k in ['label','completed','failure','summary']} for x in r.get('experiments',[])],'screen_ratios':r.get('screen_ratios')}
print(json.dumps(d))
"""
r=subprocess.run(n['SSH'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
