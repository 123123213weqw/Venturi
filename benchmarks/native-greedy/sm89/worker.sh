set -eu
OUT=/home/test/wangyue/venturi-work/qwen/greedy-validation-v3
trap 'rc=$?; echo "$rc" > "$OUT/exit-code.txt"' EXIT
cd /home/test/wangyue/venturi-work/qwen/greedy-src-v3
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
LIB=$($PY -m venturi.ops.build --component blocks --architecture sm89 --output-dir /home/test/wangyue/venturi-work/qwen/own-blocks-build)
printf '%s\n' "$LIB" > "$OUT/library-path.txt"
cp "$(dirname "$LIB")/build.json" "$OUT/build.json"
cp "$(dirname "$LIB")/build.log" "$OUT/build.log"
export VENTURI_BLOCKS_LIBRARY="$LIB"
export VENTURI_ADA_LIBRARY=/home/test/wangyue/venturi-work/qwen/own-q4-build/9842ab3e14924d72/libventuri_ada.so
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-state.txt"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/counter-policy.txt"
$PY -m pytest -q  --tb=short > "$OUT/tests.log" 2>&1 || { cat "$OUT/tests.log"; exit 1; }
tail -8 "$OUT/tests.log"
$PY tools/benchmark_greedy.py --library "$LIB" --gpu-index 0 --output-directory "$OUT/microbench" > "$OUT/microbench.log" 2>&1 || { cat "$OUT/microbench.log"; exit 1; }
cat "$OUT/microbench.log"
