"""Offline recomputation from raw native C1/C8 reports, counters and Nsight trace."""
import argparse,csv,hashlib,io,json,re,sqlite3,statistics,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def telemetry(p,started,duration):
 data=read(p);values=defaultdict(list)
 for sample in data:
  if sample.get('returncode')!=0 or not started<=sample['unix']<=started+duration:continue
  for row in csv.reader(sample['lines']):
   gpu=int(row[0]);assert row[1].strip() in ['GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28','GPU-f65db2e8-87bb-c88d-363f-23c79f793548']
   for name,x in zip(['VRAM_MiB','sampled_GPU_busy_pct','power_W','SM_MHz','memory_MHz','temperature_C'],row[2:]):values[(gpu,name)].append(float(x))
 return {str(gpu):{name:dict(min=min(xs),median=statistics.median(xs),max=max(xs),samples=len(xs)) for (g,name),xs in values.items() if g==gpu} for gpu in [0,1]}

def counter(p):
 raw=p.read_text();start=raw.find('"ID",')
 if '==ERROR==' in raw or start<0:return dict(available=False,reason=raw[:1800])
 rows=[r for r in csv.DictReader(io.StringIO(raw[start:])) if r.get('ID','').isdigit()]
 if len(rows)!=1:return dict(available=False,reason='Expected exactly one selected post-first kernel',row_count=len(rows))
 row=rows[0];result=dict(available=True,kernel=row['Kernel Name'],device=row['Device'],grid=row.get('Grid Size'),block=row.get('Block Size'))
 for k,v in row.items():
  if '__' in k:
   try:result[k]=float(v.replace(',',''))
   except (ValueError,AttributeError):result[k]=v
 duration=result.get('gpu__time_duration.sum')
 if duration:
  result['read_plus_write_GB_per_second']=(result['dram__bytes_read.sum']+result['dram__bytes_write.sum'])/duration
 return result

def trace(p):
 from importlib.util import module_from_spec,spec_from_file_location
 spec=spec_from_file_location('trace_helpers',ROOT.parent.parent/'tools/summarize_native_runtime_trace.py');helpers=module_from_spec(spec);spec.loader.exec_module(helpers)
 db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 try:
  tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")};strings=dict(db.execute('select id,value from StringIds'))
  kinds=['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']
  activities=[list(db.execute('select * from '+t)) if t in tables else [] for t in kinds]
  apis=[r for t in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'] if t in tables for r in db.execute('select * from '+t)]
  events=[e for e in db.execute('select * from NVTX_EVENTS where end is not null') if (e['text'] or strings.get(e['textId']))=='rms_native_post_first'];assert len(events)==1
  e=events[0];ids={a['correlationId'] for a in apis if a['globalTid']==e['globalTid'] and e['start']<=a['start'] and a['end']<=e['end']}
  ks,cs,ms=[[r for r in group if r['correlationId'] in ids] for group in activities];assert ks
  result=dict(scope='Profiled four-token native C8 post-first window, not scored TPS. Activity union/gaps refer to this traced process, not device-wide idle time.',host_range_ns=e['end']-e['start'],**helpers.activity_summary(ks,cs,ms,strings))
  result['by_device']=[dict(device_id=g,**helpers.activity_summary(*[[r for r in group if r['deviceId']==g] for group in [ks,cs,ms]],strings)) for g in sorted({k['deviceId'] for k in ks})]
  return result
 finally:db.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--allow-partial-transfer',action='store_true');options=p.parse_args()
 remote=ROOT/('partial-transfer' if options.allow_partial_transfer else 'remote');window=remote/'model-window';fixture=remote/'fixtures';export=remote/'export';storage=remote/'storage'
 if options.allow_partial_transfer:
  manifest=read(ROOT/'partial-transfer-manifest.json');assert manifest['complete_archive_sha256_not_verified']
  for n,h in manifest['files'].items():assert sha(remote/n)==h,n
 assert read(fixture/'completed.json')['completed'] and read(storage/'completed.json')['completed'] and read(window/'benchmark-complete.json')['completed']
 a,b=[read(window/('production-'+n+'.json')) for n in ['before','restored']]
 assert a['container_id']==b['container_id'] and a['canonical_config_fingerprint']==b['canonical_config_fingerprint'] and b['config_unchanged'] and b['healthy'] and a['counter_policy']==b['counter_policy']
 q0,q1=[read(window/f'qualify-{n}/report.json') for n in ['off','on']]
 assert q0['completed'] and q1['completed'] and q0['finite_logits_trace_state_checked'] and q1['finite_logits_trace_state_checked'] and q0['scalar_vs_batch_first_two_trajectories_equal'] and q1['scalar_vs_batch_first_two_trajectories_equal'] and q1['exact_reference_bytes_equal'] and q0['comparison_payload']==q1['comparison_payload']
 marker=read(window/'full-model-correctness-passed.json');assert sha(window/'qualify-off/report.json')==marker['baseline_sha256'] and sha(window/'qualify-on/report.json')==marker['candidate_sha256']
 builds=read(window/'build-records.json');assert builds['off']['source_sha256']==builds['on']['source_sha256']
 assert builds['off']['library_sha256']==q0['library_sha256'] and builds['on']['library_sha256']==q1['library_sha256']
 def comparable(b):return [x for x in b['command'] if not x.startswith('-DVENTURI_Q8_RMS_QUANT=')][:-1]
 assert comparable(builds['off'])==comparable(builds['on'])
 assert '-DVENTURI_Q8_RMS_QUANT=0' in builds['off']['command'] and '-DVENTURI_Q8_RMS_QUANT=1' in builds['on']['command']
 checkpoint=read(export/'export-report.json');assert q0['artifact_id']==q1['artifact_id']==checkpoint['artifact_id']
 assert checkpoint['sha256']==read(storage/'completed.json')['artifact_sha256']
 tests={}
 for name in ['build-api-tests','native-fusion-tests','producer-byte-tests','native-fallback-tests']:
  assert (fixture/(name+'.exit')).read_text().strip()=='0';tests[name]=int(re.search(r'(\d+) passed',(fixture/(name+'.log')).read_text()).group(1))
 results={};waves=[]
 for count in [1,8]:
  arms={}
  for arm in ['off','on']:
   rows=[]
   for trial in range(3):
    folder=window/f'timing-{trial}-{arm}';r=read(folder/'report.json');assert r['completed'] and not r['profiled'] and r['mode']=='timing'
    assert r['artifact_id']==q0['artifact_id'] and r['library_sha256']==(q0 if arm=='off' else q1)['library_sha256'] and r['prompt_token_ids']==q0['prompt_token_ids']
    t=next(t for t in r['trials'] if t['count']==count)
    for key,tokens in [('decode',31*count),('complete',32*count)]:
     v=t[key];assert v['output_tokens']==tokens and abs(v['tps']-tokens/v['wall_seconds'])<1e-9
    assert all(len(t['output_ids'][j])==32 for j in range(8)) if count==8 else len(t['output_ids'])==32
    rows.append(t);waves.append(dict(count=count,arm=arm,trial=trial,telemetry_decode=telemetry(folder/'gpu-telemetry.json',t['decode']['started_unix'],t['decode']['wall_seconds'])))
   arms[arm]={k:dict(median=statistics.median(row[k]['tps'] for row in rows),min=min(row[k]['tps'] for row in rows),max=max(row[k]['tps'] for row in rows),wall_seconds=[row[k]['wall_seconds'] for row in rows],process_cpu_seconds=[row[k]['process_cpu_seconds'] for row in rows],process_cpu_cores=[row[k]['process_cpu_cores'] for row in rows]) for k in ['decode','complete']}
   arms[arm]['ttft_seconds']=[row['ttft']['wall_seconds'] for row in rows]
   arms[arm]['per_request_decode_tps']=arms[arm]['decode']['median']/count
  results[str(count)]=dict(arms=arms,decode_tps_change_pct=100*(arms['on']['decode']['median']/arms['off']['decode']['median']-1),complete_tps_change_pct=100*(arms['on']['complete']['median']/arms['off']['complete']['median']-1),
   paired_decode_tps_changes_pct=[100*(next(t for t in read(window/f'timing-{trial}-on/report.json')['trials'] if t['count']==count)['decode']['tps']/next(t for t in read(window/f'timing-{trial}-off/report.json')['trials'] if t['count']==count)['decode']['tps']-1) for trial in range(3)])
 result=dict(full_evidence_archive_transferred=not options.allow_partial_transfer,evidence_integrity_scope='Full archive SHA256 verified' if not options.allow_partial_transfer else 'Only complete tar members received over authenticated SSH; local per-file SHA256 rechecked. Full remote archive hash is pinned but remaining transfer is unfinished.',scope='Qwen3.8 official source native Q8 OFF/ON, same two V100s. No MTP/HTTP/llama.cpp comparison; not production Uncensored weights. Short contexts only.',completed=True,default_enabled=False,production_candidate_deployed=False,correctness=tests,checkpoint=checkpoint,storage=read(storage/'completed.json'),configuration={k:q0[k] for k in ['artifact_id','prompt_token_ids','device_uuids','projection_profile','max_projection_rows','capacity_per_request','activation_precision','weight_storage','new_tokens','mtp']},performance=results,waves=waves,counters={},traces={},production_restored=b,profiler_repair_restored=read(remote/'timeline-repair/production-restored.json'))
 assert result['profiler_repair_restored']['healthy'] and result['profiler_repair_restored']['container_id']==a['container_id'] and result['profiler_repair_restored']['canonical_config_fingerprint']==a['canonical_config_fingerprint']
 for arm in ['off','on']:
  p=window/f'counter-{arm}.csv';result['counters'][arm]=counter(p) if p.exists() else dict(available=False,reason='No counter CSV; see profile exit/log')
  p=remote/'timeline-repair/transfer-windows'/f'timeline-{arm}.sqlite'
  if p.exists():
   pins=read(remote/'timeline-repair/trace-transfer-pins.json');assert sha(p)==pins[arm]['filtered_sha256'];result['traces'][arm]=trace(p)
  else:
   pins=read(remote/'timeline-repair/trace-transfer-pins.json');result['traces'][arm]=dict(local_raw_trace_available=False,remote_trace_pins=pins[arm],kernel_count=pins[arm]['kept_rows']['CUPTI_ACTIVITY_KIND_KERNEL'],cuda_api_count=pins[arm]['kept_rows']['CUPTI_ACTIVITY_KIND_RUNTIME'],copy_count=pins[arm]['kept_rows']['CUPTI_ACTIVITY_KIND_MEMCPY'],scope='Remote raw traces completed/pinned; local full trace transfer not yet complete. No kernel-time/gap totals inferred.')
 (ROOT/'recomputed.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(correctness=tests,performance=results,counters=result['counters'],production_restored=b['healthy']),indent=2))
if __name__=='__main__':main()
