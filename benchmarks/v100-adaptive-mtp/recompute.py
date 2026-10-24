"""Recompute routing controls and sampled telemetry from captured raw cohorts."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
raw=root/'remote/full-model'
rows=json.loads((raw/'rows.json').read_text())
samples=json.loads((raw/'gpu-samples.json').read_text())
case=json.loads((raw/'configuration.json').read_text())['case']
assert len(rows)==75
for r in rows:
 assert r['outputs']==[case['expected_generated_ids']]*r['count']
 assert r['emitted_tokens']==r['count']*case['max_new_tokens']
 assert r['wall_seconds']>0
result=[]
for count in [1,2,4,8]:
 timed=[r for r in rows if r['count']==count and r['phase'].startswith('timed-')]
 variants={}
 for v in ['auto',0,1,2]:
  trials=[r for r in timed if (r['phase']=='timed-auto' if v=='auto' else r['phase']=='timed-fixed' and r['requested_depth']==v)]
  if not trials:continue
  assert len(trials)==3
  wall=statistics.mean(r['wall_seconds'] for r in trials)
  variants[str(v)]={'mean_wall_seconds':wall,'aggregate_output_tps':trials[0]['emitted_tokens']/wall,'mean_process_cpu_seconds':statistics.mean(r['process_cpu_seconds'] for r in trials),'trial_tps':[r['aggregate_output_tps'] for r in trials],'selected_depths':[r['selected_depth'] for r in trials]}
 auto=variants['auto'];best=max((v for k,v in variants.items() if k!='auto'),key=lambda v:v['aggregate_output_tps'])
 gpu={}
 for device in [0,1]:
  values=[]
  for r in timed:
   if r['phase']!='timed-auto':continue
   for s in samples:
    if not r['start_monotonic']<=s['monotonic']<r['start_monotonic']+r['wall_seconds']:continue
    for line in s.get('csv','').splitlines():
     v=[x.strip() for x in line.split(',')]
     if v[0]==str(device):values.append([float(x) for x in v[2:]])
  gpu[str(device)]={'sample_count':len(values),'mean_gpu_busy_percent':statistics.mean(v[1] for v in values),'peak_device_used_mib':max(v[0] for v in values),'mean_power_w':statistics.mean(v[2] for v in values),'sm_clocks_mhz':sorted(set(v[3] for v in values)),'memory_clocks_mhz':sorted(set(v[4] for v in values)),'max_temperature_c':max(v[5] for v in values)}
 phases={p:{'cohorts':len(q),'wall_seconds':sum(r['wall_seconds'] for r in q),'aggregate_output_tps':sum(r['emitted_tokens'] for r in q)/sum(r['wall_seconds'] for r in q)} for p in ['cold-online-exploration','extra-calibration'] if (q:=[r for r in rows if r['count']==count and r['phase']==p])}
 result.append({'count':count,'variants':variants,'auto_gain_vs_ordinary_percent':100*(auto['aggregate_output_tps']/variants['0']['aggregate_output_tps']-1),'auto_delta_vs_best_fixed_percent':100*(auto['aggregate_output_tps']/best['aggregate_output_tps']-1),'phases':phases,'gpu':gpu})
proof={'correctness_checked_cohorts':len(rows),'checked_emitted_token_ids':sum(r['emitted_tokens'] for r in rows),'results':result,'scope':'Three-repeat global native cohort wall, not summed latencies, HTTP or post-first decode. Fixed controls use the same adaptive owner with manual overrides and learning disabled. No bare-original-API wrapper-overhead claim.'}
(root/'recomputed.json').write_text(json.dumps(proof,indent=2)+'\n')
for r in result:
 a=r['variants']['auto'];print(r['count'],a['selected_depths'],round(a['aggregate_output_tps'],4),round(r['auto_delta_vs_best_fixed_percent'],3))
