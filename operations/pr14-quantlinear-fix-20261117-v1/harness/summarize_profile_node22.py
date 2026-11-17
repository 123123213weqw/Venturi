"""Read-only Nsight timeline attribution, never throughput or SM counters."""
from pathlib import Path
import csv,hashlib,io,json,sqlite3,sys
r=Path(sys.argv[1]);e=r/'evidence'
out=dict(scope=__doc__,profile_steps=1,rows=16,tps=None,achieved_occupancy=None,dram_bandwidth_counter=None,profiles={})
for tag in ('reference','candidate'):
 p=e/f'profile-{tag}-c16.sqlite';db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 tables={v[0]for v in db.execute('select name from sqlite_master where type="table"')}
 kernels=list(db.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName'))
 total=sum(b-a for a,b,_ in kernels)
 groups={}
 for a,b,name in kernels:
  group='row_decode'if'row_dequant_rows_kernel'in name else 'gemm'if'gemm'in name.lower()else'gdn'if('gdn_recurrent'in name or'gdn_segments'in name)else'attention'if'causal_attention'in name else'other'
  g=groups.setdefault(group,dict(instances=0,total_ns=0));g['instances']+=1;g['total_ns']+=b-a
 for v in groups.values():v['kernel_time_percent']=100*v['total_ns']/total
 intervals=[(a,b)for a,b,_ in kernels]
 for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
  if table in tables:intervals+=list(db.execute('select start,end from '+table))
 intervals.sort();merged=[]
 for a,b in intervals:
  if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
  else:merged.append([a,b])
 gaps=[merged[i+1][0]-b for i,(_,b)in enumerate(merged[:-1])]
 span=merged[-1][1]-merged[0][0];covered=sum(b-a for a,b in merged)
 api={}
 if'CUPTI_ACTIVITY_KIND_RUNTIME'in tables:
  for name,count,elapsed in db.execute('SELECT s.value,count(*),sum(a.end-a.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId GROUP BY s.value'):
   if any(x in name for x in ('cudaMemcpy','cudaLaunchKernel','cudaStreamSynchronize','cudaEventSynchronize')):api[name]=dict(calls=count,total_ns=elapsed)
 rec=dict(kernel_total_ns=total,groups=groups,recorded_gpu_activity_span_ns=span,recorded_gpu_activity_covered_ns=covered,recorded_gpu_activity_coverage=covered/span,recorded_activity_gap_total_ns=span-covered,largest_recorded_activity_gaps_ns=sorted(gaps,reverse=True)[:10],cuda_api=api,raw_artifacts={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in (p,p.with_suffix('.nsys-rep'))},scope='First-to-last recorded GPU kernel/copy interval union only; not achieved SM occupancy or a counter-based utilization metric. CPU/CUDA API tail outside this interval is not included.')
 out['profiles'][tag]=rec;db.close()
try:
 policy=Path('/proc/driver/nvidia/params').read_text();out['profiling_policy']=[v.strip()for v in policy.splitlines()if'ProfilingAdminOnly'in v]
except OSError as exc:out['profiling_policy_error']=type(exc).__name__
(e/'profile-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:dict(kernel_ms=v['kernel_total_ns']/1e6,groups=v['groups'],coverage=v['recorded_gpu_activity_coverage'],gaps_ms=v['recorded_activity_gap_total_ns']/1e6,cuda_api=v['cuda_api'])for k,v in out['profiles'].items()},indent=2))
