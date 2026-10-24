import pathlib,json,sqlite3,hashlib
paths={'current-c8':'/home/data/wangyue/artifacts/venturi-wmma-final-v6/q8-residual-c8.sqlite','historical-mtp-c1':'/home/wzu/codex-build/Venturi/v100-mtp-diagnostics-v1/tp-c1/trace.sqlite'}
result={}
for label,path in paths.items():
 p=pathlib.Path(path);db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;strings=dict(db.execute('select id,value from StringIds'));events=[]
 for e in db.execute('select * from NVTX_EVENTS where end is not null'):
  name=e['text'] or strings.get(e['textId'],'')
  if name in ['native_unsplit','native_ttft','native_decode']:events.append((name,e))
 tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")};apis=[]
 for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if table in tables:apis.extend(db.execute('select * from '+table))
 mem=list(db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'));rows=[]
 for name,e in events:
  calls=[a for a in apis if a['globalTid']==e['globalTid'] and e['start']<=a['start'] and a['end']<=e['end']];ids={a['correlationId'] for a in calls};copies=[c for c in mem if c['correlationId'] in ids];twoD=[a for a in calls if 'Memcpy2DAsync' in strings[a['nameId']]]
  rows.append(dict(range=name,host_nanoseconds=e['end']-e['start'],cuda_memcpy2d_api_calls=len(twoD),cuda_memcpy2d_cpu_nanoseconds=sum(a['end']-a['start'] for a in twoD),copies={str(k):dict(count=sum(c['copyKind']==k for c in copies),bytes=sum(c['bytes'] for c in copies if c['copyKind']==k),summed_nanoseconds=sum(c['end']-c['start'] for c in copies if c['copyKind']==k)) for k in [1,2,8]}))
 result[label]=dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Read-only reuse of exact trace. Projection-primary staging is only a subset of D2D copies; API and GPU duration sums may overlap and are not elapsed-time savings or achieved bandwidth.',ranges=rows)
print(json.dumps(result,indent=2))
