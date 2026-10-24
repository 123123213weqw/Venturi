from pathlib import Path
import json
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
cap=Path('benchmarks/v100-gdn-head-pilot')
s=Path('/tmp/venturi-gdn-head-pilot-trained-v5-worker.py').read_text()
s=s.replace("out=root/'artifacts/venturi-gdn-head-pilot-trained-v5'", "out=root/'artifacts/venturi-gdn-head-pilot-profile-v6'")
pos=s.index(" run(['nvidia-smi'", s.index(" (out/'reused-payload-pins.json')"))
s=s[:pos]+r'''
 gate=root/'artifacts/venturi-gdn-head-pilot-trained-v5';assert (gate/'exit-code.txt').read_text().strip()=='0','Unprofiled trained pilot must terminate first'
 env['PYTHONPATH']+=':'+str(src/'tools')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-before.txt')
 run(['/usr/local/bin/nsys','--version'],'nsys-version.txt')
 (out/'summarize_trace.py').write_text(PARSER)
 (out/'test_trace_parser.py').write_text(TEST)
 run([py,'-m','pytest','-q',str(out/'test_trace_parser.py')],'parser-tests.log')
 cmd=[py,'tools/benchmark_gdn_head_tp.py',str(target),'--source',str(bf16),'--blocks',str(blocks),'--host-library',str(library),'--output-directory',str(out/'measurement'),'--profile']
 run(['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--cuda-event-trace=false','--cuda-graph-trace=node','-o',str(out/'gdn')]+cmd,'profile.log',timeout=1800)
 run(['/usr/local/bin/nsys','export','--type=sqlite','--output',str(out/'gdn.sqlite'),str(out/'gdn.nsys-rep')],'export.log',timeout=300)
 run([py,str(out/'summarize_trace.py'),str(out/'gdn.sqlite'),'--output',str(out/'attribution.json')],'attribution.log',timeout=300)
 report=json.loads((out/'measurement/report.json').read_text());assert report['completed'] and report['profiled'] and len(report['cells'])==3
 parsed=json.loads((out/'attribution.json').read_text());assert len(parsed['operators'])==6
 for cell in report['cells']:
  for trial in cell['trials']:
   name=f"gdn_{cell['layer']}_{cell['sequences']}x{cell['tokens_per_sequence']}_{trial['mode']}";op=next(x for x in parsed['operators'] if x['name']==name)
   for direction,short in [('host_to_device','h2d'),('device_to_host','d2h')]:
    assert op['transfers'][direction]['bytes']==trial[short+'_bytes'],name+' '+short+' bytes'
    assert op['transfers'][direction]['count']==trial[short+'_calls'],name+' '+short+' calls'
 after={str(target):identity(target)};assert after==before;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 retained={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']};(out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,profiled_separately=True,counters_matched_trace=True,no_new_cuda_kernels=True,ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
s=s.replace('try:\n expected=', 'PARSER='+repr((cap/'summarize_trace.py').read_text())+'\nTEST='+repr((cap/'test_trace_parser.py').read_text())+'\ntry:\n expected=',1)
(cap/'profile-v6-launcher.py').write_text(Path(__file__).read_text())
n['dispatch']('venturi-gdn-head-pilot-profile-v6',s)
