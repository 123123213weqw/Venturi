set -eu
cd /home/test/wangyue/venturi-work/qwen/greedy-src-v3
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=0
export VENTURI_BLOCKS_LIBRARY=/home/test/wangyue/venturi-work/qwen/own-blocks-build/393ae1122e6beb9d/libventuri_blocks.so
export VENTURI_ADA_LIBRARY=/home/test/wangyue/venturi-work/qwen/own-q4-build/9842ab3e14924d72/libventuri_ada.so
trap 'echo "$?" > /home/test/wangyue/venturi-work/qwen/greedy-validation-v3/final-tests/exit-code.txt' EXIT
/home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python -m pytest -q --tb=short
