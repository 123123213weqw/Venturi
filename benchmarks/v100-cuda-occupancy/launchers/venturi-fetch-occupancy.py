import pathlib,subprocess,json,time,shutil
repo=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=repo/'benchmarks/v100-cuda-occupancy'
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server']
check=r'''import json,pathlib
root=pathlib.Path('/home/wzu/codex-build/Venturi')
for name in ['v100-occupancy-variants-v1','v100-occupancy-prefill-v2','v100-occupancy-diagnostics-v1','v100-occupancy-final-check-v1']:
 p=root/name
 assert (p/'exit-code.txt').read_text().strip()=='0',name
 assert json.loads((p/'completed.json').read_text())['completed'],name
candidate=json.loads((root/'v100-occupancy-prefill-v1/candidate/build.json').read_text())['library_sha256']
for phase in range(4):
 for mode in ['tp','pp']:
  r=json.loads((root/f'v100-occupancy-prefill-v2/phase-{phase}-{mode}/report.json').read_text())
  assert r['completed'] and not r['profiled'] and r['new_tokens']==32
  assert r['library_sha256']==('29c2a8cc614387a9029b1cfd16dc78ebd3d9524f4083e1293eea6ea21737e6d4' if phase in (0,3) else candidate)
  assert [c['request_count'] for c in r['concurrency']]==[1,8]
for name in ['tp-c1','pp-c8']:
 r=json.loads((root/f'v100-occupancy-diagnostics-v1/{name}/profiled-run/report.json').read_text())
 assert r['completed'] and r['profiled'] and r['library_sha256']==candidate
print(json.dumps(dict(all_terminal_and_report_checks_passed=True,candidate_sha256=candidate)))
'''
r=subprocess.run(ssh+['python3 -'],input=check,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
(out/'completion-check.json').write_text(r.stdout)
route=' '.join(ssh[:-1])
for remote,local in [('v100-occupancy-v1','probe'),('v100-occupancy-variants-v1','variants'),('v100-occupancy-prefill-v1','token-aware-initial'),('v100-occupancy-prefill-v2','token-aware'),('v100-occupancy-diagnostics-v1','diagnostics'),('v100-occupancy-final-check-v1','final-check'),('v100-occupancy-postcheck-dispatch','postcheck-dispatch')]:
 d=out/'v100'/local;d.mkdir(exist_ok=True,parents=True)
 subprocess.run(['rsync','-az','-e',route,'--exclude=*.so','--exclude=*.pt','--exclude=*.sqlite','--exclude=*.nsys-rep','--exclude=__pycache__/','--exclude=*.pyc',f'WZU_Server:/home/wzu/codex-build/Venturi/{remote}/',str(d)+'/'],check=True,timeout=180)
for name in ['occupancy.json','candidate','projection-comparison','projection-comparison.log','fixture-tests.log','qualification','qualification.log']:
 source=out/'v100/token-aware-initial'/name;dest=out/'v100/token-aware'/name
 if source.is_dir():shutil.copytree(source,dest)
 else:shutil.copy2(source,dest)
subprocess.run(['python3',str(out/'summarize.py')],check=True)
