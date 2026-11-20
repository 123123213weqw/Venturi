import sqlite3,json
p='/home/data/wangyue/artifacts/venturi-native-batch-diagnostics-v6/profile-scalar.sqlite';db=sqlite3.connect('file:'+p+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
strings=dict(db.execute('select id,value from StringIds'));ks=list(db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'))
def union(xs):
 total=0;end=-1
 for a,b in sorted(xs):total+=max(0,b-max(a,end));end=max(end,b)
 return total
out={}
for key,pattern in [('decode','q8_tiled<__half, (int)1, (int)4>'),('prefill','q8_tiled<__half, (int)4, (int)1>')]:
 rows=[r for r in ks if pattern in strings[r['demangledName']]];devs=sorted({r['deviceId'] for r in rows});u={d:union([(r['start'],r['end']) for r in rows if r['deviceId']==d]) for d in devs};allunion=union([(r['start'],r['end']) for r in rows]);overlap=sum(u.values())-allunion
 out[key]=dict(kernels=len(rows),per_device_ns=u,kernel_union_ns=allunion,two_device_overlap_ns=overlap,overlap_fraction_of_union=overlap/allunion)
print(json.dumps(out,indent=2))
