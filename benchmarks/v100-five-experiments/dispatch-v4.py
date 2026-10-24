import subprocess,os,signal,json,time
from pathlib import Path
script='''import pathlib,hashlib,json,subprocess
r=pathlib.Path('/home/data/wangyue/repos/codex-build/Venturi/five-src-v4')
a=json.loads((r/'local-source-pins.json').read_text());ignored=[n for n in a if any(p.endswith('.egg-info') for p in pathlib.PurePosixPath(n).parts)]
a={n:h for n,h in a.items() if n not in ignored};bad=[n for n,h in a.items() if not (r/n).is_file() or hashlib.sha256((r/n).read_bytes()).hexdigest()!=h];assert not bad,bad
j=dict(verified_source_files=len(a),excluded_generated_package_metadata=ignored);(r/'source-validation.json').write_text(json.dumps(j,indent=2));print(json.dumps(j),flush=True)
p='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python';c='/home/data/wangyue/repos/codex-build/Venturi/five-control-v4/'
for unit,file in [('venturi-five-build-v2','build-worker-v2.py'),('venturi-five-window-v4','run-window-v4.py')]:
 subprocess.run(['systemd-run','--user','--unit='+unit,'--property=RuntimeMaxSec='+('8min' if 'build' in unit else '40min'),p,c+file],check=True)
'''
cmd=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=8','-o','ServerAliveCountMax=2','WZU_Server','python3 -']
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);t=time.time()
try:o,e=p.communicate(script.encode(),timeout=30);rc=p.returncode
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGKILL)
 for f in [p.stdin,p.stdout,p.stderr]:f.close()
 p.wait(timeout=5);o,e,rc=b'',b'outer timeout',124
j=dict(rc=rc,stdout=o.decode(),stderr=e.decode(),seconds=time.time()-t);Path(__file__).with_name('dispatch-v4-final.json').write_text(json.dumps(j,indent=2));print(json.dumps(j))
