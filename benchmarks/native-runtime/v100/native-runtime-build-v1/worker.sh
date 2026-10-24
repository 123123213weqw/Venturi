set -eu
OUT=/home/wzu/codex-build/Venturi/native-runtime-build-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/native-runtime-src-v1
export PYTHONPATH=src
/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/sm70-build > "$OUT/library-path.txt"
LIB=$(cat "$OUT/library-path.txt")
cp "$(dirname "$LIB")/build.json" "$OUT/build.json"
cp "$(dirname "$LIB")/build.log" "$OUT/build.log"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/counter-policy.txt"
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-state.txt"
