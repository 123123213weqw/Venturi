set -eu
OUT=/home/test/wangyue/venturi-work/qwen/greedy-validation-v3/profiler
trap 'rc=$?; echo "$rc" > "$OUT/exit-code.txt"' EXIT
cd /home/test/wangyue/venturi-work/qwen/greedy-src-v3
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nsys --version > "$OUT/version.txt"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/trace" /home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python tools/trace_greedy.py --library /home/test/wangyue/venturi-work/qwen/own-blocks-build/393ae1122e6beb9d/libventuri_blocks.so > "$OUT/trace.log" 2>&1
