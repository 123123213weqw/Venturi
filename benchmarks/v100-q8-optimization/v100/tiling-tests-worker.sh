set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/tiling-tests-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
BASE=/home/data/wangyue/repos/codex-build/Venturi/sm70-build/9b826970163c1fb4/libventuri_blocks.so
export VENTURI_Q8_BASELINE_LIBRARY="$BASE"
# All candidate paths are resolved from the recorded build outputs.
for NAME in t4-r1 t4-r4 t8-r4 t2-r4; do
export VENTURI_SM70_LIBRARY=$($PY -c "import json; e=next(e for e in json.load(open('$OUT/tiling-libraries.json')) if 't'+str(e['token_tile'])+'-r'+str(e['row_tile'])=='$NAME'); print(e['library'])")
$PY -m pytest -q tests/test_q8_tiling.py --tb=short > "$OUT/tiling-tests-$NAME.log" 2>&1
done
export VENTURI_SM70_LIBRARY=$($PY -c "import json; print(next(e['library'] for e in json.load(open('$OUT/tiling-libraries.json')) if e['token_tile']==4 and e['row_tile']==4))")
$PY -m pytest -q tests/test_qwen_native.py tests/test_fp16_port.py tests/test_fp32_parameters.py tests/test_sampling.py tests/test_ada_build.py tests/test_qwen_native_resident_tool.py --tb=short > "$OUT/tiling-hybrid-tests.log" 2>&1
$PY tools/summarize_native_runtime_trace.py "$OUT/baseline-profiler/trace.sqlite" --output "$OUT/baseline-profiler/attribution.json"
$PY -u tools/benchmark_q8_tiles.py /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --baseline "$BASE" --candidates "$OUT/tiling-libraries.json" --output-directory "$OUT/projection-ablation" --gpu-index 1 > "$OUT/projection-ablation.log" 2>&1
