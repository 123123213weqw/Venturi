set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-validation-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_PP_TEST=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
$PY -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/v100-pp-build > "$OUT/library.txt"
export VENTURI_SM70_LIBRARY=$(cat "$OUT/library.txt")
cp "$(dirname "$VENTURI_SM70_LIBRARY")/build.json" "$OUT/build.json"
cp "$(dirname "$VENTURI_SM70_LIBRARY")/build.log" "$OUT/build.log"
$PY -m pytest -x -q tests/test_qwen_pipeline.py tests/test_qwen_tensor_parallel.py tests/test_qwen_native.py tests/test_fp16_port.py tests/test_ada_build.py tests/test_residency.py --tb=short > "$OUT/tests.log" 2>&1
