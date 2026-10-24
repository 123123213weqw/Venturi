"""NVML samples inside unprofiled scored calls; never substitute them for counters."""
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'evidence/window';result=[]
for p in sorted(D.glob('timing-*/report.json')):
 j=json.loads(p.read_text());samples=json.loads(p.with_name('gpu-telemetry.json').read_text())
 for trial in j['trials']:
  for scope in ['decode','complete']:
   t=trial[scope];selected=[s for s in samples if t['started_unix']<=s['unix']<=t['started_unix']+t['wall_seconds'] and s.get('returncode')==0]
   by_device={}
   for device in [0,1]:
    rows=[]
    for s in selected:
     for line in s['lines']:
      v=[x.strip() for x in line.split(',')]
      if int(v[0])==device:rows.append([float(x) for x in v[2:]])
    if not rows:by_device[device]=dict(samples=0);continue
    keys=['memory_MiB','gpu_busy_pct','power_W','sm_MHz','mem_MHz','temperature_C'];by_device[device]=dict(samples=len(rows),**{k:dict(min=min(v[i] for v in rows),max=max(v[i] for v in rows),mean=statistics.mean(v[i] for v in rows)) for i,k in enumerate(keys)})
   result.append(dict(name=p.parent.name,mask=j['args']['mask'],trial=j['args']['trial'],count=trial['count'],scope=scope,by_device=by_device))
(R/'telemetry.json').write_text(json.dumps(result,indent=2)+'\n')
for x in result:
 if x['count']==8 and x['scope']=='decode':print(x['name'],json.dumps(x['by_device']))
