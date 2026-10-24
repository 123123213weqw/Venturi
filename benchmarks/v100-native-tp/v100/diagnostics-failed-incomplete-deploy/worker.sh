set -eu
OUT=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tp-tools-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-tp-validation-v2/library.txt)
export VENTURI_SM70_LIBRARY="$LIB"
$PY -m pip install --only-binary=:all: --no-deps --target "$OUT/python-deps" --retries 2 --timeout 30 numpy==1.26.4 > "$OUT/dependency-install.log" 2>&1
export PYTHONPATH="$OUT/python-deps:src:tests"
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
CASE=/home/wzu/codex-build/Venturi/streaming-validation/case.json
$PY -m pytest -q --tb=short > "$OUT/tests.log" 2>&1
$PY "$OUT/peer-probe.py" > "$OUT/peer-probe.json" 2> "$OUT/peer-probe.log"
for MODE in single parallel; do
mkdir "$OUT/$MODE-profiler"
if test "$MODE" = parallel; then
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/$MODE-profiler/trace" "$PY" tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/$MODE-profiler/profiled-run" --new-tokens 8 --repeats 1 --dtype fp16 --weight-mode resident --gpu-index 1 --profile-nvtx --tensor-parallel-device cuda:1 --tensor-parallel-transport host --peer-gpu-index 0 > "$OUT/$MODE-profiler/trace.log" 2>&1
else
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/$MODE-profiler/trace" "$PY" tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/$MODE-profiler/profiled-run" --new-tokens 8 --repeats 1 --dtype fp16 --weight-mode resident --gpu-index 1 --profile-nvtx > "$OUT/$MODE-profiler/trace.log" 2>&1
fi
$PY tools/summarize_native_runtime_trace.py "$OUT/$MODE-profiler/trace.sqlite" --output "$OUT/$MODE-profiler/attribution.json"
done
$PY -m venturi qwen-native generate "$MODEL" --blocks-library "$LIB" --weight-mode resident --dtype fp16 --token-ids 248045,846,198,3710,369,279,6511,314,9338,30,21134,440,279,3177,803,1132,13,248046,198,248045,74455,198,248068,271,248069,271 --max-new-tokens 2 --max-tokens 26 --eos-token-ids 248046,248044 --tensor-parallel-device cuda:1 --tensor-parallel-transport host > "$OUT/cli-smoke.json" 2> "$OUT/cli-smoke.log"
nvidia-smi topo -m > "$OUT/topology.txt"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/hardware-counter-policy.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
