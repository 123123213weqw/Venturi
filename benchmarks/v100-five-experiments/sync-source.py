import subprocess,os,signal,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
remote='/home/data/wangyue/repos/codex-build/Venturi/five-src-v1/'
cmd=['rsync','-az','--timeout=45','--exclude=.git/','--exclude=target/','--exclude=node_modules/','--exclude=.env','--exclude=.env.*','--exclude=__pycache__/','--exclude=.pytest_cache/','--exclude=*.egg-info/','-e','ssh -o BatchMode=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes -o ConnectTimeout=12','--relative','src','tests','tools','docs','AGENTS.md','pyproject.toml','README.md','WZU_Server:'+remote]
t=time.time();p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
try:o,e=p.communicate(timeout=180);rc=p.returncode
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGKILL)
 for x in [p.stdout,p.stderr]:x.close()
 p.wait(timeout=5);o,e,rc=b'',b'outer timeout',124
j=dict(command=cmd,rc=rc,stdout=o.decode(errors='replace'),stderr=e.decode(errors='replace'),seconds=time.time()-t)
Path(__file__).with_name('sync-source.json').write_text(json.dumps(j,indent=2));print(json.dumps(j),flush=True)
