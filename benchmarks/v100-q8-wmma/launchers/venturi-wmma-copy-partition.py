import pathlib,sqlite3,json,hashlib
out={}
for count in [1,8]:
 p=pathlib.Path(f'/home/data/wangyue/artifacts/venturi-wmma-final-v6/q8-residual-c{count}.sqlite');db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;strings=dict(db.execute('select id,value from StringIds'));e=next(e for e in db.execute('select * from NVTX_EVENTS where end is not null') if (e['text'] or strings.get(e['textId']))=='native_unsplit');calls=[]
 for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  try:calls.extend(a for a in db.execute('select * from '+table) if a['globalTid']==e['globalTid'] and e['start']<=a['start'] and a['end']<=e['end'])
  except sqlite3.OperationalError:pass
 kinds={c['correlationId']:c['copyKind'] for c in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')};groups={}
 for a in calls:
  name=strings[a['nameId']]
  if 'Memcpy' not in name:continue
  key=f"{name}/kind={kinds.get(a['correlationId'],'unjoined')}";d=groups.setdefault(key,dict(count=0,api_nanoseconds=0));d['count']+=1;d['api_nanoseconds']+=a['end']-a['start']
 out[str(count)]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Read-only API/copy-correlation partition. CPU API durations may contain GPU dependency waits; not all time is removable overhead and sums are not an elapsed-time speedup prediction.',groups=groups)
print(json.dumps(out,indent=2))
