set -eu
OUT=/home/wzu/codex-build/Venturi/v100-warp-experiment-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-warp-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY "$OUT/build.py"
export VENTURI_Q8_BASELINE_LIBRARY=/home/data/wangyue/repos/codex-build/Venturi/v100-q8-tiling-build/78bf7865cd726a8a/libventuri_blocks.so
for MODE in 1 2; do
export VENTURI_SM70_LIBRARY=$($PY -c "import json; print(next(e['library'] for e in json.load(open('$OUT/libraries.json')) if e['decode_mode']==$MODE))")
$PY -m pytest -q tests/test_q8_tiling.py tests/test_qwen_native.py --tb=short > "$OUT/tests-mode-$MODE.log" 2>&1
done
$PY -u tools/benchmark_q8_tiles.py /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --baseline "$VENTURI_Q8_BASELINE_LIBRARY" --candidates "$OUT/libraries.json" --output-directory "$OUT/ablation" --gpu-index 1 --decode-only > "$OUT/ablation.log" 2>&1
