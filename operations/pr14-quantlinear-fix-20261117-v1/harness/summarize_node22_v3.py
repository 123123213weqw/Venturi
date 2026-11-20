"""Derive qualified medians from complete clean repetitions; no GPU work."""
from pathlib import Path
import json,statistics,sys
R=Path(sys.argv[1]);E=R/'evidence'
q=json.loads((E/'sm70-real-v1-report.json').read_text());assert q['completed'] and q['passed']
t=json.loads((E/'tps-matched-v1.json').read_text());assert t['completed']
samples_by_arm={arm:[json.loads(x)for x in ((Path('/home/xs/codex-build/Venturi-pr14-final-20261117-v2/evidence') if arm=='dequant' else E)/'sm70-tps-v1-gpu.jsonl').read_text().splitlines()] for arm in ('dequant','auto')}
out=dict(artifact_sha256=q['artifact_sha256'],host='V100_222/root1; two V100-PCIE-32GB native replicas',scope=t['scope'],tokenizer_streaming_ttft_measured=False,hardware_occupancy_and_dram_bandwidth_measured=False,quality=dict(bf16_or_corpus_qualified=q['original_bf16_or_corpus_quality_qualified'],cases=len(q['implementations']['auto']['cases']),batch_counts=[x['concurrency']for x in q['batches']],all_batch_tokens_equal=all(x['tokens_equal']for x in q['batches']),state_failures=sum(len(x['state_failures'])for x in q['batches'])),implementations={})
out['reference_reuse']=t.get('reference_reuse')
for arm,data in t['implementations'].items():
 samples=samples_by_arm[arm]
 assert data['completed']
 cells=[]
 for c in (1,4,8,16,32):
  rows=[r for r in data['runs']if r['concurrency']==c];assert len(rows)==3
  assert all(r['valid'] and r['aggregate_tps'] is not None and r['qualified_output_ids_equal'] and r['gpu_process_window']['clean'] for r in rows)
  times=[x['time_to_first_token_seconds']for r in rows for x in r['scheduler_timings']if x]
  gpu={}
  for sample in samples:
   if not any(r['unix']<=sample['timestamp']<=r['unix']+r['interval_s'] for r in rows):continue
   for text in sample['data']:
    index,uuid,busy,memory,power,sm,memclock,temp=[x.strip()for x in text.split(',')]
    gpu.setdefault(uuid,[]).append(list(map(float,(busy,memory,power,sm,memclock,temp))))
  cells.append(dict(concurrency=c,repetitions=3,median_tps=statistics.median(r['aggregate_tps']for r in rows),min_tps=min(r['aggregate_tps']for r in rows),max_tps=max(r['aggregate_tps']for r in rows),median_response_latency_s=statistics.median(r['response_latency_median_s']for r in rows),median_scheduler_first_token_availability_s=statistics.median(times)if times else None,median_server_cpu_seconds=statistics.median(sum(r['server_cpu_seconds'])for r in rows),median_client_and_observer_cpu_seconds=statistics.median(r['client_cpu_seconds']for r in rows),median_server_cpu_wall_ratio=statistics.median(sum(r['server_cpu_seconds'])/r['interval_s']for r in rows),max_server_rss_gib=[max(r['server_rss_kib'][i]for r in rows)/1024**2 for i in (0,1)],gpu={uuid:dict(samples=len(v),mean_busy_percent=statistics.mean(x[0]for x in v),peak_vram_mib=max(x[1]for x in v),mean_power_w=statistics.mean(x[2]for x in v),sm_clock_mhz_range=[min(x[3]for x in v),max(x[3]for x in v)],memory_clock_mhz_range=[min(x[4]for x in v),max(x[4]for x in v)],temperature_c_range=[min(x[5]for x in v),max(x[5]for x in v)])for uuid,v in gpu.items()}))
 out['implementations'][arm]=dict(cells=cells,discarded_contaminated_runs=len(data.get('discarded_contaminated_runs',[])))
out['observed_median_delta_percent']={c['concurrency']:100*(c['median_tps']/r['median_tps']-1)for r,c in zip(out['implementations']['dequant']['cells'],out['implementations']['auto']['cells'])}
(E/'summary-qualified.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
