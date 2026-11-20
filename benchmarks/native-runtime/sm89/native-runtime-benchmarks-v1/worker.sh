set -eu
OUT=/home/test/wangyue/venturi-work/qwen/native-runtime-benchmarks-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/test/wangyue/venturi-work/qwen/native-runtime-src-v1
export PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=0
PY=/home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python
LIB=$(cat /home/test/wangyue/venturi-work/qwen/native-runtime-build-v1/library-path.txt)
export VENTURI_BLOCKS_LIBRARY="$LIB"
$PY -m pytest -q tests/test_qwen_native_cli.py tests/test_qwen_native_tool.py > "$OUT/extra-tests.log" 2>&1
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-state.txt"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/counter-policy.txt"
$PY - <<'FIXTURE'
from pathlib import Path
import json
from test_qwen_model import hybrid_fixture
root=Path('/home/test/wangyue/venturi-work/qwen/native-runtime-benchmarks-v1/fixture');root.mkdir();weights,_=hybrid_fixture(root)
print(weights.artifact.path)
Path('/home/test/wangyue/venturi-work/qwen/native-runtime-benchmarks-v1/fixture-case.json').write_text(json.dumps(dict(prompt_token_ids=[1,4,3,2,1,8,7,6],max_new_tokens=8,eos_token_ids=[]),indent=2)+'\n')
FIXTURE
for MODE in resident streaming; do
$PY -u tools/benchmark_qwen_native.py "$OUT/fixture/hybrid-q8.ninfer" --library "$LIB" --case "$OUT/fixture-case.json" --output-directory "$OUT/fixture-$MODE" --weight-mode "$MODE" --dtype bf16 --gpu-index 0 --max-tokens 8 --repeats 2 --inner-requests 20 --order python,native,python,native --scope 'Synthetic four-layer Qwen hybrid fixture, NOT 27B performance' > "$OUT/fixture-$MODE.log" 2>&1
done
mkdir "$OUT/profiler"
nsys --version > "$OUT/profiler/version.txt"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/profiler/trace" "$PY" tools/benchmark_qwen_native.py "$OUT/fixture/hybrid-q8.ninfer" --library "$LIB" --case "$OUT/fixture-case.json" --output-directory "$OUT/profiler/profiled-fixture-resident" --weight-mode resident --dtype bf16 --gpu-index 0 --max-tokens 8 --repeats 2 --inner-requests 1 --order python,native --scope 'Profiler-only fixture; NOT unprofiled timing or 27B performance' --profile-nvtx > "$OUT/profiler/trace.log" 2>&1
