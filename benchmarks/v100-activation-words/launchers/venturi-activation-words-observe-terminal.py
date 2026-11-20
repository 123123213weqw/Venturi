from pathlib import Path
import subprocess,runpy
code='''import pathlib,json,time
r=pathlib.Path('/home/data/wangyue/artifacts');gate=r/'venturi-activation-words-guarded-final-v10';deadline=time.monotonic()+1800
while not (gate/'exit-code.txt').exists():
 pid=json.loads((gate/'process.json').read_text())['pid'];c=pathlib.Path(f'/proc/{pid}/cmdline');assert c.exists() and str(gate/'worker.py').encode() in c.read_bytes(),'Pinned final handle absent; inspect predecessor failure, never relaunch'
 for n in ['guarded-model-v8','guarded-campaign-v9']:
  p=r/('venturi-activation-words-'+n)
  if (p/'exit-code.txt').exists():assert (p/'exit-code.txt').read_text().strip()=='0',(p/'failure.txt').read_text() if (p/'failure.txt').exists() else 'Predecessor failure'
 assert time.monotonic()<deadline,'Observation timeout only; inspect same live handle'
 time.sleep(30)
assert (gate/'exit-code.txt').read_text().strip()=='0',(gate/'failure.txt').read_text() if (gate/'failure.txt').exists() else 'Final worker failed'
print('All guarded jobs completed successfully.')
p=r/'venturi-activation-words-guarded-campaign-v9';d=json.loads((p/'campaign-report.json').read_text());print(json.dumps({k:d[k] for k in ['completed','screen_ratios','path_promotions']},indent=2))
print(json.dumps([dict(label=e['label'],summary=e['summary']) for e in d['experiments'] if e['label'].startswith('repeat-')],indent=2))
'''
r=subprocess.run(runpy.run_path('/tmp/venturi-dispatch-helper.py')['SSH'],input=code,text=True,capture_output=True,timeout=1830);print(r.stdout,r.stderr);r.check_returncode();Path('/tmp/venturi-activation-words-terminal-observation.txt').write_text(r.stdout)
