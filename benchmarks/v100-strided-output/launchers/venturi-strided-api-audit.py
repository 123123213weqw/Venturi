import pathlib,sqlite3,json,hashlib
r=pathlib.Path('/home/data/wangyue/artifacts');out={}
for label,folder in [('control','venturi-wmma-final-v6'),('candidate','venturi-strided-final-v8')]:
 for c in [1,8]:
  p=r/folder/f'q8-residual-c{c}.sqlite';db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;strings=dict(db.execute('select id,value from StringIds'));events=[e for e in db.execute('select * from NVTX_EVENTS where end is not null') if (e['text'] or strings.get(e['textId']))=='native_unsplit'];groups={};tables={a[0] for a in db.execute("select name from sqlite_master where type='table'")}
  for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
   if table not in tables:continue
   for a in db.execute('select * from '+table):
    if not any(a['globalTid']==e['globalTid'] and e['start']<=a['start'] and a['end']<=e['end'] for e in events):continue
    key=table+'/'+strings[a['nameId']];item=groups.setdefault(key,dict(count=0,api_nanoseconds=0));item['count']+=1;item['api_nanoseconds']+=a['end']-a['start']
  out[f'{label}-c{c}']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Submitting-thread native range only. Runtime/driver API durations may include nested calls and GPU waits, and are not additive elapsed-time savings.',groups=groups);db.close()
print(json.dumps(out,indent=2))
