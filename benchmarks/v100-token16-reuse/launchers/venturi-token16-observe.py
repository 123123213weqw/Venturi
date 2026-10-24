from pathlib import Path
import runpy,subprocess,json
code='''import pathlib,time,json,statistics
r=pathlib.Path('/home/data/wangyue/artifacts');p=r/'venturi-token16-study-v1';deadline=time.monotonic()+900
while not (p/'exit-code.txt').exists():
 pid=json.loads((p/'process.json').read_text())['pid'];c=pathlib.Path(f'/proc/{pid}/cmdline');assert c.exists() and str(p/'worker.py').encode() in c.read_bytes(),'Pinned study handle absent';assert time.monotonic()<deadline,'Observation timeout, inspect same handle, never restart';time.sleep(20)
rc=int((p/'exit-code.txt').read_text());assert rc==0,(p/'failure.txt').read_text() if (p/'failure.txt').exists() else rc
op=json.loads((p/'operators/report.json').read_text());ratio=lambda c:c['median_gpu_event_ms']['control-t8']/c['median_gpu_event_ms']['candidate-t16']
d=dict(library=(p/'library.txt').read_text().strip(),sha256=op['candidate_sha256'],cases=len(op['cases']),median_eligible_fused_ratio=statistics.median(ratio(c) for c in op['cases'] if c['candidate_native_tile16_eligible'] and c['label'].startswith('fused-')),median_eligible_plain_ratio=statistics.median(ratio(c) for c in op['cases'] if c['candidate_native_tile16_eligible'] and not c['label'].startswith('fused-')),minimum_unchanged_fused_t8_ratio=min(ratio(c) for c in op['cases'] if c['tokens']==8 and c['label'].startswith('fused-')),fused_t16=[dict(label=c['label'],tokens=c['tokens'],ms=c['median_gpu_event_ms']) for c in op['cases'] if c['tokens']==16 and c['label'].startswith('fused-')])
f=json.loads((p/'candidate-resources.json').read_text());d['target_resources']=[v for v in f['devices'][0]['compiled_q8_functions'] if any(x in v['symbol'] for x in ['3vdt3dotILi16ELi2ELi8E','3vqf9tiled_dotILi16ELi2ELi8E','3vdt3dotILi8ELi2ELi16E','3vqf9tiled_dotILi8ELi2ELi16E'])]
print(json.dumps(d,indent=2))
'''
r=subprocess.run(runpy.run_path('/tmp/venturi-dispatch-helper.py')['SSH'],input=code,text=True,capture_output=True,timeout=930);print(r.stdout,r.stderr);r.check_returncode();Path('benchmarks/v100-token16-reuse/operator-observation.json').write_text(r.stdout)
