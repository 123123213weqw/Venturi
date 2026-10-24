set -eu
OUT=/home/wzu/codex-build/Venturi/native-runtime-tests-v2
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/native-runtime-src-v1
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=1
LIB=$(cat /home/wzu/codex-build/Venturi/native-runtime-build-v1/library-path.txt)
export VENTURI_SM70_LIBRARY="$LIB"
/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python -m pytest -q tests/test_qwen_native.py tests/test_fp16_port.py tests/test_fp32_parameters.py tests/test_sampling.py tests/test_ada_build.py --tb=short
