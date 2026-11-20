"""Profiled host API durations, not CPU compute time or an unprofiled speed result.

Runtime/driver categories may nest and must not be added together. Synchronize
API duration can include GPU waiting. CUDA device identifiers are trace-local.
"""
import json
from pathlib import Path
import sqlite3
import sys


def summarize(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        strings=dict(db.execute('select id,value from StringIds'))
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        result=[]
        for e in db.execute('select * from NVTX_EVENTS where end is not null'):
            label=e['text'] or strings.get(e['textId'])
            if label not in ('native_ttft','native_decode','native_unsplit'):continue
            families={}
            for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
                if table not in tables:continue
                rows=list(db.execute(f'select * from {table} where globalTid=? and start>=? and end<=?',(e['globalTid'],e['start'],e['end'])))
                grouped={}
                for a in rows:
                    name=strings[a['nameId']];v=grouped.setdefault(name,dict(count=0,nanoseconds=0))
                    v['count']+=1;v['nanoseconds']+=a['end']-a['start']
                families[table]=sorted([dict(name=name,**v) for name,v in grouped.items()],key=lambda x:x['nanoseconds'],reverse=True)
            result.append(dict(name=label,host_range_nanoseconds=e['end']-e['start'],families=families))
        assert result
        return dict(scope=__doc__,ranges=result)
    finally:db.close()


if __name__=='__main__':
    p=Path(sys.argv[1]);(p/'api-attribution.json').write_text(json.dumps(summarize(p/'trace.sqlite'),indent=2)+'\n')
