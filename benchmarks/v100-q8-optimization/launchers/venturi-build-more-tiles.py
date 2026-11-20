import subprocess,json
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
code='''import pathlib,json,subprocess,time
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
assert not (p/'more-build-process.json').exists()
code="""import json,pathlib\nfrom venturi.ops.build import build_blocks_backend\np=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')\nr=json.load((p/'tiling-libraries.json').open())\nfor row in (2,8):\n lib=build_blocks_backend('/home/wzu/codex-build/Venturi/v100-q8-tiling-build',architecture='sm70',q8_token_tile=4,q8_row_tile=row)\n r.append(dict(token_tile=4,row_tile=row,library=str(lib),manifest=json.load(lib.with_name('build.json').open())))\n (p/'more-tiling-libraries.json').write_text(json.dumps(r,indent=2)+'\\n')\n print(row,lib,flush=True)\n"""
(p/'more-build.py').write_text(code)
script="""set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/more-build-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export PYTHONPATH=src
/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python "$OUT/more-build.py"
"""
(p/'more-build.sh').write_text(script)
with (p/'more-build.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(p/'more-build.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'more-build-process.json').write_text(json.dumps(r));print(json.dumps(r))
'''
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
