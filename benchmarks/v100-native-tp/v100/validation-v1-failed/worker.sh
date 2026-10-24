set -eu
OUT=/home/wzu/codex-build/Venturi/v100-tp-validation-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tp-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_TP_TEST=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/v100-tp-build > "$OUT/library.txt"
LIB=$(cat "$OUT/library.txt")
export VENTURI_SM70_LIBRARY="$LIB"
cp "$(dirname "$LIB")/build.json" "$OUT/build.json"
cp "$(dirname "$LIB")/build.log" "$OUT/build.log"
$PY -m pytest -q tests/test_qwen_tensor_parallel.py tests/test_qwen_native.py tests/test_fp16_port.py tests/test_fp32_parameters.py tests/test_sampling.py tests/test_ada_build.py --tb=short > "$OUT/tests.log" 2>&1
