"""Post-process raw unprofiled trials separately from Nsys/NCU evidence."""
import argparse,csv,io,json,sqlite3,statistics
from collections import defaultdict
from pathlib import Path

def union(intervals):
 total=0;end=None
 for a,b in sorted(intervals):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total

def timeline(p):
 c=sqlite3.connect(p);tables={x[0] for x in c.execute("select name from sqlite_master where type='table'")};names=dict(c.execute('select id,value from StringIds'))
 rows=list(c.execute('select start,end,deviceId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'))
 copies=list(c.execute('select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY'))
 activities=defaultdict(list);kernel_ms=defaultdict(float);kernel_n=defaultdict(int)
 for a,b,d,n in rows:activities[d].append((a,b));kernel_ms[names[n]]+=(b-a)/1e6;kernel_n[names[n]]+=1
 cp=defaultdict(lambda:dict(count=0,bytes=0,summed_gpu_ms=0))
 labels=dict(c.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
 for a,b,d,k,n in copies:
  activities[d].append((a,b));v=cp[str(d)+':'+labels.get(k,str(k))];v['count']+=1;v['bytes']+=n;v['summed_gpu_ms']+=(b-a)/1e6
 if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
  for a,b,d in c.execute('select start,end,deviceId from CUPTI_ACTIVITY_KIND_MEMSET'):activities[d].append((a,b))
 first=min(a for q in activities.values() for a,b in q);last=max(b for q in activities.values() for a,b in q);span=last-first
 api=defaultdict(lambda:dict(count=0,summed_host_ms=0))
 for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if table not in tables:continue
  for a,b,n in c.execute('select start,end,nameId from '+table):v=api[names[n]];v['count']+=1;v['summed_host_ms']+=(b-a)/1e6
 result=dict(scope='Four profiled C8 post-first steps, NOT unprofiled TPS; idle means no recorded kernel/copy/memset on a common two-device activity span',kernel_count=len(rows),copy_count=len(copies),span_ms=span/1e6,
  devices={str(d):dict(recorded_activity_ms=union(q)/1e6,kernel_union_ms=union([(a,b) for a,b,dev,n in rows if dev==d])/1e6,idle_fraction=1-union(q)/span) for d,q in activities.items()},copies=dict(cp),
  kernel_summed_ms={n:dict(count=kernel_n[n],ms=t) for n,t in sorted(kernel_ms.items(),key=lambda x:-x[1])},api=dict(api),driver_table_present='CUPTI_ACTIVITY_KIND_DRIVER' in tables)
 c.close();return result

def counters(p):
 rows=list(csv.reader(io.StringIO(p.read_text())));start=next(i for i,r in enumerate(rows) if r and r[0]=='ID');header,units,values=rows[start:start+3]
 units,values=dict(zip(header,units)),dict(zip(header,values));metrics={}
 for key,v in values.items():
  if key.startswith(('gpu__','sm__','dram__','l1tex__','lts__','smsp__')):
   try:metrics[key]=dict(value=float(v.replace(',','')),unit=units.get(key,''))
   except ValueError:metrics[key]=dict(raw=v,unit=units.get(key,''))
 duration=metrics['gpu__time_duration.sum'];scale={'nsecond':1e-9,'ns':1e-9,'usecond':1e-6,'us':1e-6,'msecond':1e-3,'ms':1e-3,'second':1}.get(duration['unit']);assert scale,duration
 seconds=duration['value']*scale
 bandwidth=(metrics['dram__bytes_read.sum']['value']+metrics['dram__bytes_write.sum']['value'])/seconds/1e9
 return dict(scope='One C8 causal_attention kernel on primary GPU only; not whole-model occupancy/bandwidth or unprofiled TPS; caches/clocks uncontrolled',kernel=values.get('Kernel Name'),device=values.get('Device'),block=values.get('Block Size'),grid=values.get('Grid Size'),metrics=metrics,derived_dram_read_write_GBps=bandwidth)

def kv_delta(before,after):
 return {k:after[k]-before.get(k,0) for k in ['evictions','reloads','device_to_host_bytes','host_to_device_bytes','allocation_ooms'] if k in after}

def window(d):
 out=dict(path=str(d),timings=[],qualification={},context={},timeline={},counters={});group=defaultdict(list)
 for p in sorted(d.glob('*/report.json')):
  r=json.loads(p.read_text());name=p.parent.name
  if name.startswith('timing'):
   variant=r['args']['kv_mode'];variant='spill' if r['args']['kv_budget_mib'] else variant
   for t in r['trials']:
    row=dict(name=name,variant=variant,count=t['count'],decode_tps=t['decode']['tps'],complete_tps=t['complete']['tps'],ttft_seconds=t['ttft']['wall_seconds'],decode_wall=t['decode']['wall_seconds'],decode_cpu=t['decode']['process_cpu_seconds'],decode_cpu_cores=t['decode']['process_cpu_cores'],kv_delta=kv_delta(t['kv_before_decode'],t['kv_after_decode']),info=t['info'])
    out['timings'].append(row);group[(variant,t['count'])].append(row)
  elif name.startswith('qualify') or name.startswith('capacity'):
   out['qualification'][name]=dict(completed=r['completed'],exact_reference_bytes_equal=r.get('exact_reference_bytes_equal'),capacity=r['capacity_per_request'],prompt_tokens=len(r['prompt_token_ids'][0]),output_tokens=r['new_tokens'],kv_cache=r['kv_cache'])
  elif name.startswith('context-'):
   out['context'][name]={k:r[k] for k in ['completed','mode','prompt_tokens','output_tokens','batch','capacity','prefill','decode','kv_before_decode','kv_after_decode','max_rss_kib']};out['context'][name]['exact_reference_bytes_equal']=r.get('exact_reference_bytes_equal')
 for p in sorted(d.glob('timeline-*.sqlite')):out['timeline'][p.stem]=timeline(p)
 for p in sorted(d.glob('counter-*.csv')):
  try:out['counters'][p.stem]=counters(p)
  except Exception as e:out['counters'][p.stem]=dict(parse_error=repr(e))
 out['aggregates']={f'{v}/C{n}':dict(repeats=len(rows),decode_mean=statistics.mean(x['decode_tps'] for x in rows),decode_median=statistics.median(x['decode_tps'] for x in rows),decode_range=[min(x['decode_tps'] for x in rows),max(x['decode_tps'] for x in rows)],complete_mean=statistics.mean(x['complete_tps'] for x in rows),mean_decode_cpu_cores=statistics.mean(x['decode_cpu_cores'] for x in rows)) for (v,n),rows in group.items()}
 # NVML samples are diagnostics, not occupancy or DRAM utilization.
 telemetry=defaultdict(list)
 for p in sorted(d.glob('timing*/gpu-telemetry.json')):
  samples=json.loads(p.read_text());report=json.loads((p.parent/'report.json').read_text())
  for t in report['trials']:
   start=t['decode']['started_unix'];end=start+t['decode']['wall_seconds']
   for s in samples:
    if not start<=s['unix']<=end:continue
    for line in s.get('lines',[]):
     fields=[x.strip() for x in line.split(',')]
     try:values=[float(x) for x in fields[2:]];assert len(values)==6
     except (ValueError,AssertionError):continue
     telemetry[(p.parent.name,t['count'],fields[0])].append(values)
 out['nvml_decode']={f'{name}/C{count}/GPU{device}':dict(samples=len(rows),**{key:dict(mean=statistics.mean(r[i] for r in rows),min=min(r[i] for r in rows),max=max(r[i] for r in rows)) for i,key in enumerate(['vram_MiB','gpu_busy_pct','power_W','sm_clock_MHz','memory_clock_MHz','temperature_C'])}) for (name,count,device),rows in telemetry.items()}
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result={n:window(a.evidence/n) for n in ['window-v2','window-final-v3'] if (a.evidence/n).exists()}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({n:r['aggregates'] for n,r in result.items()},indent=2))
