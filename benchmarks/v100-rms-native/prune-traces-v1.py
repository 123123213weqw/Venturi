"""Keep exact raw rows for the post-first NVTX window; pin complete originals remotely.

No durations, kernel names or counters are modified. Transfer reduction only:
source Nsight SQLite/rep remain on the V100 host with recorded SHA256.
"""
import hashlib,json,sqlite3
from pathlib import Path
ROOT=Path('/home/data/wangyue/artifacts/venturi-rms-timeline-repair-v1')

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*2**20),b''):h.update(b)
 return h.hexdigest()

def main():
 assert json.loads((ROOT/'completed.json').read_text())['completed'] and json.loads((ROOT/'production-restored.json').read_text())['healthy']
 out=ROOT/'transfer-windows';out.mkdir(exist_ok=False);record={}
 for arm in ['off','on']:
  source=ROOT/('timeline-'+arm+'.sqlite');target=out/source.name
  db=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
  dest=sqlite3.connect(target)
  try:
   schemas=dict(db.execute("select name,sql from sqlite_master where type='table'"));strings=dict(db.execute('select id,value from StringIds'))
   events=[e for e in db.execute('select * from NVTX_EVENTS where end is not null') if (e['text'] or strings.get(e['textId']))=='rms_native_post_first'];assert len(events)==1;e=events[0]
   def keep(table,rows):
    dest.execute(schemas[table]);rows=list(rows)
    if rows:dest.executemany('insert into "'+table+'" values ('+','.join('?' for _ in rows[0])+')',[tuple(r) for r in rows])
    return len(rows)
   counts={};counts['StringIds']=keep('StringIds',db.execute('select * from StringIds'));counts['NVTX_EVENTS']=keep('NVTX_EVENTS',events)
   ids=set()
   for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
    if table in schemas:
     rows=[a for a in db.execute('select * from '+table) if a['globalTid']==e['globalTid'] and e['start']<=a['start'] and a['end']<=e['end']]
     ids.update(a['correlationId'] for a in rows);counts[table]=keep(table,rows)
   for table in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
    if table in schemas:counts[table]=keep(table,[r for r in db.execute('select * from '+table) if r['correlationId'] in ids])
   assert counts['CUPTI_ACTIVITY_KIND_KERNEL']>0;dest.commit()
  finally:dest.close();db.close()
  record[arm]=dict(source=str(source),source_sha256=digest(source),source_bytes=source.stat().st_size,nsys_rep_sha256=digest(ROOT/('timeline-'+arm+'.nsys-rep')),filtered_sha256=digest(target),filtered_bytes=target.stat().st_size,kept_rows=counts,scope=__doc__)
 (ROOT/'trace-transfer-pins.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
