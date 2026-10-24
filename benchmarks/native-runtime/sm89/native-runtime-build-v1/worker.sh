set -eu
OUT=/home/test/wangyue/venturi-work/qwen/native-runtime-build-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/test/wangyue/venturi-work/qwen/native-runtime-src-v1
export PYTHONPATH=src
/home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python -m venturi.ops.build --component blocks --architecture sm89 --output-dir /home/test/wangyue/venturi-work/qwen/own-blocks-build > "$OUT/library-path.txt"
LIB=$(cat "$OUT/library-path.txt")
cp "$(dirname "$LIB")/build.json" "$OUT/build.json"
cp "$(dirname "$LIB")/build.log" "$OUT/build.log"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/counter-policy.txt"
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-state.txt"
