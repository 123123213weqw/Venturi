from pathlib import Path
import subprocess,json
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
wait='''import pathlib,time,json
p=pathlib.Path('/home/data/wangyue/artifacts/venturi-batched-mtp-sdk-final-v9');deadline=time.monotonic()+1200
while not (p/'exit-code.txt').exists():
 pid=json.loads((p/'process.json').read_text())['pid'];c=pathlib.Path(f'/proc/{pid}/cmdline');assert c.exists() and str(p/'worker.py').encode() in c.read_bytes(),'Pinned profiler worker absent';assert time.monotonic()<deadline,'Observation deadline reached, revalidate existing handle without relaunch';time.sleep(10)
assert (p/'exit-code.txt').read_text().strip()=='0',(p/'failure.txt').read_text() if (p/'failure.txt').exists() else 'Failed final stage'
time.sleep(1)
'''
r=subprocess.run(ns['SSH'],input=wait+Path('/tmp/venturi-batched-mtp-terminal.py').read_text(),text=True,capture_output=True,timeout=1250)
if r.returncode:
 print(r.stderr);raise SystemExit(r.returncode)
d=json.loads(r.stdout);Path('benchmarks/v100-batched-mtp/terminal-audit.json').write_text(json.dumps(d,indent=2)+'\n');print('Terminal audit:',len(d['workers']),'owned workers, original GPU footprint restored')
