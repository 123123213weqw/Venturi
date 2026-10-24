set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/baseline-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-resident-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=/home/data/wangyue/repos/codex-build/Venturi/sm70-build/9b826970163c1fb4/libventuri_blocks.so
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
CASE=/home/wzu/codex-build/Venturi/streaming-validation/case.json
export VENTURI_SM70_LIBRARY="$LIB"
$PY -m pytest -q tests/test_qwen_native.py tests/test_qwen_native_cli.py tests/test_qwen_native_resident_tool.py > "$OUT/baseline-tests.log" 2>&1
$PY -u tools/qualify_qwen_native.py "$MODEL" --expected-sha256 b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8 --blocks-library "$LIB" --previous-directory /home/wzu/codex-build/Venturi/greedy-validation-v3/full-model-run1 --previous-report-sha256 cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732 --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --dtype fp16 --weight-mode resident --output-directory "$OUT/resident-qualification" > "$OUT/resident-qualification.log" 2>&1
$PY -u tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/resident-baseline" --new-tokens 32 --repeats 3 --dtype fp16 --weight-mode resident --gpu-index 1 > "$OUT/resident-baseline.log" 2>&1
mkdir "$OUT/baseline-profiler"
nsys --version > "$OUT/baseline-profiler/version.txt"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/baseline-profiler/trace" "$PY" tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/baseline-profiler/profiled-run" --new-tokens 8 --repeats 1 --dtype fp16 --weight-mode resident --gpu-index 1 --profile-nvtx > "$OUT/baseline-profiler/trace.log" 2>&1
