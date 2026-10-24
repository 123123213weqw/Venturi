import pathlib,subprocess,json
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
import shlex
r=subprocess.run(['rsync','-az',*[f'--exclude={x}' for x in ['.git/','target/','node_modules/','.env','.env.*','__pycache__/','.pytest_cache/','*.pem','*.key']],'-e',shlex.join(ssh),'src','tests','tools','pyproject.toml','AGENTS.md','WZU_Server:/home/wzu/codex-build/Venturi/v100-tiling-src-v1/'],cwd=root,check=True)
build='''import json,pathlib
from venturi.ops.build import build_blocks_backend
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
r=[]
for t,row in [(4,1),(4,4),(8,4),(2,4)]:
 lib=build_blocks_backend('/home/wzu/codex-build/Venturi/v100-q8-tiling-build',architecture='sm70',q8_token_tile=t,q8_row_tile=row)
 r.append(dict(token_tile=t,row_tile=row,library=str(lib),manifest=json.load(lib.with_name('build.json').open())))
 (p/'tiling-libraries.json').write_text(json.dumps(r,indent=2)+'\\n')
 print(t,row,lib,flush=True)
'''
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/tiling-build-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export PYTHONPATH=src
/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python "$OUT/tiling-build.py"
'''
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
assert not (out/'tiling-build-process.json').exists()
(out/'tiling-build.py').write_text({build!r})
(out/'tiling-build.sh').write_text({worker!r})
with (out/'tiling-build.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(out/'tiling-build.sh')],cwd='/home/wzu/codex-build/Venturi/v100-tiling-src-v1',stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'tiling-build-process.json').write_text(json.dumps(r));print(json.dumps(r))
'''
r=subprocess.run(ssh+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60)
print(r.stdout,r.stderr);r.check_returncode()
