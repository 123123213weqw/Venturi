set -eu
OUT=/home/wzu/codex-build/Venturi/v100-tp-full-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tp-src-v2
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-tp-validation-v2/library.txt)
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
CASE=/home/wzu/codex-build/Venturi/streaming-validation/case.json
$PY -u tools/qualify_qwen_native.py "$MODEL" --expected-sha256 b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8 --blocks-library "$LIB" --previous-directory /home/wzu/codex-build/Venturi/greedy-validation-v3/full-model-run1 --previous-report-sha256 cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732 --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --dtype fp16 --weight-mode resident --tensor-parallel-device cuda:1 --tensor-parallel-transport host --output-directory "$OUT/qualification" > "$OUT/qualification.log" 2>&1
for PHASE in 0 1 2 3; do
if test "$PHASE" = 0 || test "$PHASE" = 3; then
$PY -u tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/matched-phase-$PHASE" --new-tokens 32 --repeats 2 --dtype fp16 --weight-mode resident --gpu-index 1 > "$OUT/matched-phase-$PHASE.log" 2>&1
else
$PY -u tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/matched-phase-$PHASE" --new-tokens 32 --repeats 2 --dtype fp16 --weight-mode resident --gpu-index 1 --tensor-parallel-device cuda:1 --tensor-parallel-transport host --peer-gpu-index 0 > "$OUT/matched-phase-$PHASE.log" 2>&1
fi
done
for PHASE in 0 1; do
if test "$PHASE" = 0; then
$PY -u tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$OUT/story-case.json" --output-directory "$OUT/story-phase-$PHASE" --new-tokens 32 --repeats 2 --dtype fp16 --weight-mode resident --gpu-index 1 > "$OUT/story-phase-$PHASE.log" 2>&1
else
$PY -u tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$OUT/story-case.json" --output-directory "$OUT/story-phase-$PHASE" --new-tokens 32 --repeats 2 --dtype fp16 --weight-mode resident --gpu-index 1 --tensor-parallel-device cuda:1 --tensor-parallel-transport host --peer-gpu-index 0 > "$OUT/story-phase-$PHASE.log" 2>&1
fi
done
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
