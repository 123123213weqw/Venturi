"""Summarize only a cudaProfilerStart/Stop decode capture, never use it for TPS."""
import importlib.util,json,sqlite3,hashlib,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('trace_helpers',Path(__file__).resolve().parents[2]/'tools/summarize_native_runtime_trace.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def summarize(p):
 db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 try:
  tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")};strings=dict(db.execute('select id,value from StringIds'))
  def rows(n):return list(db.execute('select * from '+n)) if n in tables else []
  if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables and db.execute('select count(*) from CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()[0]:raise ValueError('Graph-level trace omits kernel nodes; use --cuda-graph-trace=node')
  ks=rows('CUPTI_ACTIVITY_KIND_KERNEL');cs=rows('CUPTI_ACTIVITY_KIND_MEMCPY');ms=rows('CUPTI_ACTIVITY_KIND_MEMSET');apis=rows('CUPTI_ACTIVITY_KIND_RUNTIME')+rows('CUPTI_ACTIVITY_KIND_DRIVER')
  assert ks and len({r['globalPid'] for r in ks})==1
  devices=sorted({h.device_id(x) for x in ks+cs+ms});names={}
  for a in apis:
   n=strings[a['nameId']];names[n]=names.get(n,0)+1
  return dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cuda_calls=len(apis),api_counts=names,**h.activity_summary(ks,cs,ms,strings),by_device=[dict(device=d,**h.activity_summary([x for x in ks if h.device_id(x)==d],[x for x in cs if h.device_id(x)==d],[x for x in ms if h.device_id(x)==d],strings)) for d in devices],scope=__doc__+' Activity union gaps describe this process only, not device-wide idle time. Device duration sums can overlap. Instrumented scheduling is not unprofiled behavior.')
 finally:db.close()
if __name__=='__main__':
 p=Path(sys.argv[1]);r=summarize(p);Path(sys.argv[2]).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['by_device','kernel_by_name','api_counts']},indent=2))
