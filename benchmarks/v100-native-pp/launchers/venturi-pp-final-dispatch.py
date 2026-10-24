import subprocess,pathlib,hashlib,json
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
sha=json.loads((root/'benchmarks/v100-native-tp/v100/full/qualification/report.json').read_text())['artifact_sha256']
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-final-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-final-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_PP_TEST=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/v100-pp-build > "$OUT/library.txt"
LIB=$(cat "$OUT/library.txt");export VENTURI_SM70_LIBRARY="$LIB"
cp "$(dirname "$LIB")/build.json" "$OUT/build.json"
cp "$(dirname "$LIB")/build.log" "$OUT/build.log"
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
$PY -m pytest -x -q tests/test_qwen_pipeline.py tests/test_qwen_tensor_parallel.py tests/test_qwen_native.py --tb=short > "$OUT/fixture-tests.log" 2>&1
$PY -u tools/qualify_qwen_native.py "$MODEL" --expected-sha256 @SHA@ --blocks-library "$LIB" --previous-directory /home/wzu/codex-build/Venturi/greedy-validation-v3/full-model-run1 --previous-report-sha256 cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732 --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --dtype fp16 --weight-mode resident --pipeline-device cuda:1 --pipeline-split-layer 32 --output-directory "$OUT/qualification" > "$OUT/qualification.log" 2>&1
$PY -m pytest -q --tb=short > "$OUT/tests.log" 2>&1
cp /home/wzu/codex-build/Venturi/v100-pp-comparison-v1/case.json "$OUT/case.json"
for PHASE in 0 1 2 3; do
if test "$PHASE" = 0 || test "$PHASE" = 3; then MODE=tp; else MODE=pp; fi
$PY -u tools/benchmark_qwen_native_parallel.py "$MODEL" --library "$LIB" --case "$OUT/case.json" --mode "$MODE" --output-directory "$OUT/phase-$PHASE" --concurrency 1,2,4,8 --repeats 1 --new-tokens 32 --split-layer 32 > "$OUT/phase-$PHASE.log" 2>&1
done
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
'''.replace('@SHA@',sha)
files=['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/ops/csrc/qwen_pipeline.cuh','src/venturi/models/qwen_native.py','src/venturi/models/qwen_pipeline.py','tools/benchmark_qwen_native_parallel.py','tools/qualify_qwen_native.py']
expected={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
code=f'''import pathlib,subprocess,time,json,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-final-v1');p.mkdir()
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-final-src-v1');expected={expected!r}
assert all(hashlib.sha256((src/f).read_bytes()).hexdigest()==h for f,h in expected.items())
(p/'worker.sh').write_text({worker!r})
waiter=\"\"\"import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-comparison-v1/exit-code.txt')
for _ in range(120):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Initial comparison did not finish within bounded wait; no target jobs started')
assert q.read_text().strip()=='0','Initial comparison failed'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-pp-final-v1/worker.sh']))
\"\"\"
(p/'waiter.py').write_text(waiter)
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'waiter.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256=expected,scheduled_after='Initial ABBA terminal success; complete correctness repeated before final timing')
(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(pid=proc.pid)))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
