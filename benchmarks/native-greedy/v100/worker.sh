set -eu
OUT=/home/wzu/codex-build/Venturi/greedy-validation-v3
trap 'rc=$?; echo "$rc" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/greedy-src-v3
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
export CUDA_VISIBLE_DEVICES=1
LIB=$($PY -m venturi.ops.build --component blocks --architecture sm70 --output-dir /home/wzu/codex-build/Venturi/sm70-build)
printf '%s\n' "$LIB" > "$OUT/library-path.txt"
cp "$(dirname "$LIB")/build.json" "$OUT/build.json"
cp "$(dirname "$LIB")/build.log" "$OUT/build.log"
export VENTURI_SM70_LIBRARY="$LIB"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-state.txt"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/counter-policy.txt"
$PY -m pytest -q tests/test_sampling.py tests/test_qwen_streaming.py tests/test_qwen_streaming_tool.py tests/test_fp32_parameters.py tests/test_fp16_port.py tests/test_ada_build.py --tb=short > "$OUT/tests.log" 2>&1 || { cat "$OUT/tests.log"; exit 1; }
tail -8 "$OUT/tests.log"
$PY tools/benchmark_greedy.py --library "$LIB" --gpu-index 1 --output-directory "$OUT/microbench" > "$OUT/microbench.log" 2>&1 || { cat "$OUT/microbench.log"; exit 1; }
cat "$OUT/microbench.log"
$PY -u tools/qualify_qwen_streaming.py /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --blocks-library "$LIB" --expected-sha256 b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8 --case /home/wzu/codex-build/Venturi/streaming-validation/case.json --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --output-directory "$OUT/full-model-run1" --dtype fp16 --sampler native-greedy --run-cpu-reference > "$OUT/full-model.log" 2>&1
