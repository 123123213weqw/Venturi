set -eu
OUT=/home/test/wangyue/venturi-work/qwen/native-runtime-tests-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/test/wangyue/venturi-work/qwen/native-runtime-src-v1
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=0
LIB=$(cat /home/test/wangyue/venturi-work/qwen/native-runtime-build-v1/library-path.txt)
export VENTURI_BLOCKS_LIBRARY="$LIB"
export VENTURI_ADA_LIBRARY=/home/test/wangyue/venturi-work/qwen/own-q4-build/9842ab3e14924d72/libventuri_ada.so
/home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python -m pytest -q  --tb=short
