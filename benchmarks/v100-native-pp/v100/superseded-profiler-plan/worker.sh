set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-diagnostics-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-analysis-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_PP_TEST=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-pp-validation-v2/library.txt)
export VENTURI_SM70_LIBRARY="$LIB"
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
CASE=/home/wzu/codex-build/Venturi/v100-pp-comparison-v1/case.json
$PY -m pytest -q --tb=short > "$OUT/tests.log" 2>&1
for MODE in tp pp; do
for COUNT in 1 8; do
D="$OUT/$MODE-c$COUNT";mkdir "$D"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$D/trace" "$PY" tools/benchmark_qwen_native_parallel.py "$MODEL" --library "$LIB" --case "$CASE" --mode "$MODE" --output-directory "$D/profiled-run" --concurrency "$COUNT" --repeats 1 --new-tokens 8 --split-layer 32 --profile-nvtx > "$D/trace.log" 2>&1
$PY tools/summarize_native_parallel_trace.py "$D/trace.sqlite" --output "$D/cohort-attribution.json"
done
done
$PY -m venturi qwen-native generate "$MODEL" --blocks-library "$LIB" --weight-mode resident --dtype fp16 --token-ids 248045,846,198,3710,369,279,6511,314,9338,30,21134,440,279,3177,803,1132,13,248046,198,248045,74455,198,248068,271,248069,271 --max-new-tokens 2 --max-tokens 26 --eos-token-ids 248046,248044 --pipeline-device cuda:1 --pipeline-split-layer 32 > "$OUT/cli-smoke.json" 2> "$OUT/cli-smoke.log"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/hardware-counter-policy.txt"
nvidia-smi topo -m > "$OUT/topology.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
