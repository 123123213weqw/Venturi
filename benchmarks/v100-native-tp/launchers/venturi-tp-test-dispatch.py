import subprocess
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-tp-validation-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tp-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_TP_TEST=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/v100-tp-build > "$OUT/library.txt"
LIB=$(cat "$OUT/library.txt")
export VENTURI_SM70_LIBRARY="$LIB"
cp "$(dirname "$LIB")/build.json" "$OUT/build.json"
cp "$(dirname "$LIB")/build.log" "$OUT/build.log"
$PY -m pytest -q tests/test_qwen_tensor_parallel.py tests/test_qwen_native.py tests/test_fp16_port.py tests/test_fp32_parameters.py tests/test_sampling.py tests/test_ada_build.py --tb=short > "$OUT/tests.log" 2>&1
'''
code=f'''import pathlib,json,subprocess,time,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-tp-validation-v1');p.mkdir()
(p/'worker.sh').write_text({worker!r})
with (p/'worker.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(p/'worker.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-tp-src-v1')
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256={{f:hashlib.sha256((src/f).read_bytes()).hexdigest() for f in ['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/models/qwen_native.py']}})
(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
