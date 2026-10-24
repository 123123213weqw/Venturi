import subprocess
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
build='''import json,pathlib
from venturi.ops.build import build_blocks_backend
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
r=json.load((p/'tiling-libraries.json').open())
for row in (2,8):
 lib=build_blocks_backend('/home/wzu/codex-build/Venturi/v100-q8-tiling-build',architecture='sm70',q8_token_tile=4,q8_row_tile=row)
 r.append(dict(token_tile=4,row_tile=row,library=str(lib),manifest=json.load(lib.with_name('build.json').open())))
 with (p/'more-tiling-libraries-v2.json').open('w') as f:json.dump(r,f,indent=2)
 print(row,lib,flush=True)
'''
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/more-build-v2-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export PYTHONPATH=src
/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python "$OUT/more-build-v2.py"
'''
code=f'''import pathlib,subprocess,time,json
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
assert (p/'more-build-exit-code.txt').read_text().strip()=='1'
assert not (p/'more-build-v2-process.json').exists()
(p/'more-build-v2.py').write_text({build!r});(p/'more-build-v2.sh').write_text({worker!r})
with (p/'more-build-v2.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(p/'more-build-v2.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'more-build-v2-process.json').write_text(json.dumps(r));print(json.dumps(r))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
