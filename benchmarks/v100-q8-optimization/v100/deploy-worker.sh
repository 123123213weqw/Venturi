set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/deploy-exit-code.txt"' EXIT
for ATTEMPT in $(seq 1 30); do
if test -f "$OUT/final-exit-code.txt"; then break; fi
sleep 10
done
test "$(cat "$OUT/final-exit-code.txt")" = 0
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/v100-q8-tiling-build > "$OUT/default-library.txt"
LIB=$(cat "$OUT/default-library.txt")
test "$LIB" = /home/data/wangyue/repos/codex-build/Venturi/v100-q8-tiling-build/78bf7865cd726a8a/libventuri_blocks.so
export VENTURI_SM70_LIBRARY="$LIB"
export VENTURI_Q8_BASELINE_LIBRARY=/home/data/wangyue/repos/codex-build/Venturi/sm70-build/9b826970163c1fb4/libventuri_blocks.so
$PY -m pytest -q tests/test_qwen_native.py tests/test_fp16_port.py tests/test_fp32_parameters.py tests/test_sampling.py tests/test_ada_build.py tests/test_qwen_native_resident_tool.py tests/test_q8_tiling.py tests/test_native_runtime_trace.py tests/test_qwen_native_cli.py tests/test_qwen_native_tool.py --tb=short > "$OUT/final-tests.log" 2>&1
IDS=$($PY -c 'import json; print(",".join(map(str,json.load(open("/home/wzu/codex-build/Venturi/streaming-validation/case.json"))["prompt_token_ids"])))')
$PY -m venturi qwen-native generate /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --blocks-library "$LIB" --weight-mode resident --dtype fp16 --token-ids "$IDS" --max-new-tokens 2 --max-tokens 26 --eos-token-ids 248046,248044 > "$OUT/cli-smoke.json" 2> "$OUT/cli-smoke.stderr"
$PY -c "import json; d=json.load(open('$OUT/cli-smoke.json')); assert d['generated_ids']==[57590,248046] and d['native_memory']['position']==27"
/usr/local/cuda/bin/cuobjdump --dump-resource-usage "$LIB" > "$OUT/static-resources.txt"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/counter-policy-final.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
docker inspect --format "name={{.Name}} pid={{.State.Pid}} running={{.State.Running}}" fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d > "$OUT/service-final.txt"
$PY -c "import time,json; print(json.dumps(dict(local_timezone=time.tzname,utc_offset_seconds=time.localtime().tm_gmtoff,observed_unix=time.time())))" > "$OUT/server-clock.json"
