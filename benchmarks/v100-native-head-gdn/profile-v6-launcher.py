from pathlib import Path
import json
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);cap=Path('benchmarks/v100-native-head-gdn')
old=Path('/tmp/venturi-native-head-gdn-mtp-v4-worker.py').read_text();source=old[:old.index('try:\n expected=')].replace("out=root/'artifacts/venturi-native-head-gdn-mtp-v4'", "out=root/'artifacts/venturi-native-head-gdn-profile-v6'")
expected=json.loads((cap/'source-manifest-v3.json').read_text())
source+='try:\n expected='+repr(expected)+r'''
 campaign=root/'artifacts/venturi-native-head-gdn-campaign-v5';deadline=time.monotonic()+5700
 while not (campaign/'exit-code.txt').exists():
  pid=json.loads((campaign/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(campaign/'worker.py').encode() in q.read_bytes(),'Campaign handle missing without terminal marker';assert time.monotonic()<deadline,'Verified live campaign still pending; do not restart';time.sleep(5)
 assert (campaign/'exit-code.txt').read_text().strip()=='0' and json.loads((campaign/'campaign-report.json').read_text())['completed']
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def identity(p):
  s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 prior=root/'artifacts/venturi-native-head-gdn-mtp-v4';library=pathlib.Path((prior/'library.txt').read_text().strip());assert sha(library)=='33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8'
 build=json.loads(library.with_name('build.json').read_text());assert build['library_sha256']==sha(library) and len(build['source_sha256'])==11
 for rel,h in build['source_sha256'].items():assert sha(src/'src/venturi/ops/csrc'/rel)==h
 (out/'library.txt').write_text(str(library)+chr(10))
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json'
 before={str(p):identity(p) for p in [target,draft]};assert before==json.loads((campaign/'payload-after.json').read_text());(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 traces=[]
 for arm,profile in [('control','q8-residual-column-mlp'),('candidate','q8-residual-column-mlp-head-gdn')]:
  for mode in ['mtp2-c1','mtp-batch-c8']:
   name=arm+'-'+mode
   if mode=='mtp2-c1':cmd=[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--case',str(case),'--depth','2','--count','1','--recovery','prefix']
   else:cmd=[py,'tools/profile_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--case',str(case),'--count','8']
   cmd+=['--projection-profile',profile,'--mtp-draft-projection-profile','q8-residual','--output',str(out/(name+'-metadata.json'))]
   run(['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--cuda-event-trace=false','--cuda-graph-trace=node','-o',str(out/name)]+cmd,name+'-profile.log',timeout=1200)
   run(['/usr/local/bin/nsys','export','--type=sqlite','--output',str(out/(name+'.sqlite')),str(out/(name+'.nsys-rep'))],name+'-export.log',timeout=300)
   run([py,'tools/summarize_native_parallel_trace.py',str(out/(name+'.sqlite')),'--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=300)
   traces += ['--trace',name+'='+str(out/(name+'.sqlite'))]
 run([py,'tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',sha(library),'--output',str(out/'resources.json')]+traces,'resources.log',timeout=600)
 after={str(p):identity(p) for p in [target,draft]};assert before==after;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 retained={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']};(out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library_sha256=sha(library),separate_from_unprofiled=True,achieved_hardware_counters=None,ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
(cap/'profile-v6-launcher.py').write_text(Path(__file__).read_text())
n['dispatch']('venturi-native-head-gdn-profile-v6',source)
