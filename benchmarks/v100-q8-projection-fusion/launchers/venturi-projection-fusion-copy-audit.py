import pathlib,json,sqlite3,hashlib
root=pathlib.Path('/home/data/wangyue/artifacts');base=root/'venturi-projection-fusion-final-v5'
paths={name:base/(name+'.sqlite') for name in ['control-c1','control-c8','grouped-c1','grouped-c8','candidate-c1','candidate-c8','candidate-mtp-c1']}
result={};partitions={}
for label,p in paths.items():
 db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;strings=dict(db.execute('select id,value from StringIds'));events=[e for e in db.execute('select * from NVTX_EVENTS where end is not null') if (e['text'] or strings.get(e['textId']))=='native_unsplit'];tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")};apis=[]
 for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if table in tables:apis.extend(db.execute('select * from '+table))
 mem=list(db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'));rows=[];groups={}
 for e in events:
  calls=[a for a in apis if a['globalTid']==e['globalTid'] and e['start']<=a['start'] and a['end']<=e['end']];ids={a['correlationId'] for a in calls};copies=[c for c in mem if c['correlationId'] in ids];kinds={c['correlationId']:c['copyKind'] for c in copies};twoD=[a for a in calls if 'Memcpy2DAsync' in strings[a['nameId']]]
  rows.append(dict(range='native_unsplit',host_nanoseconds=e['end']-e['start'],cuda_memcpy2d_api_calls=len(twoD),cuda_memcpy2d_cpu_nanoseconds=sum(a['end']-a['start'] for a in twoD),copies={str(k):dict(count=sum(c['copyKind']==k for c in copies),bytes=sum(c['bytes'] for c in copies if c['copyKind']==k),summed_nanoseconds=sum(c['end']-c['start'] for c in copies if c['copyKind']==k)) for k in [1,2,8]}))
  for a in calls:
   name=strings[a['nameId']]
   if 'Memcpy' not in name:continue
   key=f"{name}/kind={kinds.get(a['correlationId'],'unjoined')}";d=groups.setdefault(key,dict(count=0,api_nanoseconds=0));d['count']+=1;d['api_nanoseconds']+=a['end']-a['start']
 meta=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Read-only API/copy-correlation attribution restricted to the submitting-thread native range. CPU API sums can contain dependency waits and nested API time; device copies can overlap. Neither is an elapsed-time savings or achieved-bandwidth measurement.')
 result[label]=dict(**meta,ranges=rows);partitions[label]=dict(**meta,groups=groups);db.close()
(base/'copy-audit.json').write_text(json.dumps(result,indent=2)+'\n');(base/'copy-api-partition.json').write_text(json.dumps(partitions,indent=2)+'\n')
