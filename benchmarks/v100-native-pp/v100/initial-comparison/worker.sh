set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-comparison-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-execution-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-pp-validation-v2/library.txt)
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
for PHASE in 0 1 2 3; do
if test "$PHASE" = 0 || test "$PHASE" = 3; then MODE=tp; else MODE=pp; fi
$PY -u tools/benchmark_qwen_native_parallel.py "$MODEL" --library "$LIB" --case "$OUT/case.json" --mode "$MODE" --output-directory "$OUT/phase-$PHASE" --concurrency 1,2,4,8 --repeats 1 --new-tokens 32 --split-layer 32 > "$OUT/phase-$PHASE.log" 2>&1
done
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
