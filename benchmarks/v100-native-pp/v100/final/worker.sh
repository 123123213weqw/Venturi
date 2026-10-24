set -eu
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
$PY -u tools/qualify_qwen_native.py "$MODEL" --expected-sha256 b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8 --blocks-library "$LIB" --previous-directory /home/wzu/codex-build/Venturi/greedy-validation-v3/full-model-run1 --previous-report-sha256 cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732 --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --dtype fp16 --weight-mode resident --pipeline-device cuda:1 --pipeline-split-layer 32 --output-directory "$OUT/qualification" > "$OUT/qualification.log" 2>&1
$PY -m pytest -q --tb=short > "$OUT/tests.log" 2>&1
cp /home/wzu/codex-build/Venturi/v100-pp-comparison-v1/case.json "$OUT/case.json"
for PHASE in 0 1 2 3; do
if test "$PHASE" = 0 || test "$PHASE" = 3; then MODE=tp; else MODE=pp; fi
$PY -u tools/benchmark_qwen_native_parallel.py "$MODEL" --library "$LIB" --case "$OUT/case.json" --mode "$MODE" --output-directory "$OUT/phase-$PHASE" --concurrency 1,2,4,8 --repeats 1 --new-tokens 32 --split-layer 32 > "$OUT/phase-$PHASE.log" 2>&1
done
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
