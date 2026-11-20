import subprocess,pathlib,hashlib
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-validation-v2
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-src-v2
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_PP_TEST=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/v100-pp-build > "$OUT/library.txt"
export VENTURI_SM70_LIBRARY=$(cat "$OUT/library.txt")
cp "$(dirname "$VENTURI_SM70_LIBRARY")/build.json" "$OUT/build.json"
cp "$(dirname "$VENTURI_SM70_LIBRARY")/build.log" "$OUT/build.log"
$PY -m pytest -x -q tests/test_qwen_pipeline.py tests/test_qwen_tensor_parallel.py tests/test_qwen_native.py tests/test_fp16_port.py tests/test_ada_build.py tests/test_residency.py --tb=short > "$OUT/tests.log" 2>&1
'''
code=f'''import pathlib,subprocess,time,json,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-validation-v2');p.mkdir()
(p/'worker.sh').write_text({worker!r})
with (p/'worker.log').open('xb') as log: proc=subprocess.Popen(['bash',str(p/'worker.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-src-v2')
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256={{str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest() for part in ['src','tests','tools','scripts'] for f in (src/part).rglob('*') if f.is_file() and '__pycache__' not in f.parts}})
(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(pid=proc.pid,source_files=len(r['source_sha256']))))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
