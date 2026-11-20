import subprocess
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
build='''import pathlib,json
from venturi.ops.build import build_blocks_backend
out=pathlib.Path('/home/wzu/codex-build/Venturi/v100-warp-experiment-v1');r=[]
for mode in (1,2):
 lib=build_blocks_backend('/home/wzu/codex-build/Venturi/v100-warp-build',architecture='sm70',q8_token_tile=4,q8_row_tile=4,q8_decode_mode=mode)
 r.append(dict(name='warp'+str(mode),token_tile=4,row_tile=4,decode_mode=mode,library=str(lib),manifest=json.load(lib.with_name('build.json').open())))
 (out/'libraries.json').write_text(json.dumps(r,indent=2))
 print(mode,lib,flush=True)
'''
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-warp-experiment-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-warp-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY "$OUT/build.py"
export VENTURI_Q8_BASELINE_LIBRARY=/home/data/wangyue/repos/codex-build/Venturi/v100-q8-tiling-build/78bf7865cd726a8a/libventuri_blocks.so
for MODE in 1 2; do
export VENTURI_SM70_LIBRARY=$($PY -c "import json; print(next(e['library'] for e in json.load(open('$OUT/libraries.json')) if e['decode_mode']==$MODE))")
$PY -m pytest -q tests/test_q8_tiling.py tests/test_qwen_native.py --tb=short > "$OUT/tests-mode-$MODE.log" 2>&1
done
$PY -u tools/benchmark_q8_tiles.py /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --baseline "$VENTURI_Q8_BASELINE_LIBRARY" --candidates "$OUT/libraries.json" --output-directory "$OUT/ablation" --gpu-index 1 --decode-only > "$OUT/ablation.log" 2>&1
'''
code=f'''import pathlib,subprocess,time,json
out=pathlib.Path('/home/wzu/codex-build/Venturi/v100-warp-experiment-v1');out.mkdir()
(out/'build.py').write_text({build!r});(out/'worker.sh').write_text({worker!r})
with (out/'worker.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(out/'worker.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r));print(json.dumps(r))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
