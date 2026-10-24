import pathlib,json,sqlite3,hashlib,collections
r=pathlib.Path('/home/data/wangyue/artifacts');out={}
for folder,names in [
 ('venturi-token-order-final-v7',['mtp2-prefix-c1','ordinary256-c8','mtp-batch-c8']),
 ('venturi-native-mtp2-full-profile-v14',['mtp2-prefix-full-c1']),
 ('venturi-batched-mtp-final-v4',['ordinary256-control-c8']),
 ('venturi-batched-mtp-sdk-final-v9',['mtp-batch-c8'])]:
 for name in names:
  p=r/folder/(name+'.sqlite');db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
  strings={x['id']:x['value'] for x in db.execute('select * from StringIds')};ranges=[]
  for e in db.execute('select * from NVTX_EVENTS where end is not null'):
   label=e['text'] or strings.get(e['textId'])
   if label in ['native_unsplit','native_ttft','native_decode']:ranges.append(e)
  ids={a['correlationId'] for a in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME') if any(a['globalTid']==e['globalTid'] and e['start']<=a['start'] and a['end']<=e['end'] for e in ranges)}
  groups=collections.defaultdict(lambda:dict(launches=0,kernel_nanoseconds=0))
  for k in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
   n=strings[k['demangledName']]
   if k['correlationId'] not in ids or not any(s in n for s in ['vqf::','vdt::']):continue
   fields=('deviceId','gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory','localMemoryPerThread')
   key=(n,*(k[s] for s in fields));g=groups[key];g['launches']+=1;g['kernel_nanoseconds']+=k['end']-k['start']
  rows=[dict(zip(('name',*fields),k),**v) for k,v in groups.items()];assert rows
  out[folder+'/'+name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_nvtx_ranges=len(ranges),projection_launch_groups=rows)
  db.close()
print(json.dumps(dict(scope='Read-only CUPTI/NVTX launch geometry and compiled trace registers, not achieved occupancy/cache/DRAM counters. Grid dimensions retain actual X/Y rather than only their product; fresh candidate and explicitly reused controls.',traces=out),indent=2))
