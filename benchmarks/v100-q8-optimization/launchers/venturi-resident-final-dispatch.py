import subprocess,json,pathlib
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
identity=json.load((root/'benchmarks/native-runtime/v100/native-runtime-validation/full-model-v1/report.json').open())
subprocess.run(['scp']+opts+[str(root/'benchmarks/v100-q8-optimization/story-case.json'),'WZU_Server:/home/wzu/codex-build/Venturi/v100-resident-optimization-v1/story-case.json'],check=True,timeout=60)
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/final-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
BASE=/home/data/wangyue/repos/codex-build/Venturi/sm70-build/9b826970163c1fb4/libventuri_blocks.so
TILED=/home/data/wangyue/repos/codex-build/Venturi/v100-q8-tiling-build/78bf7865cd726a8a/libventuri_blocks.so
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
CASE=/home/wzu/codex-build/Venturi/streaming-validation/case.json
$PY -u tools/qualify_qwen_native.py "$MODEL" --expected-sha256 @ARTIFACT_SHA@ --blocks-library "$TILED" --previous-directory /home/wzu/codex-build/Venturi/greedy-validation-v3/full-model-run1 --previous-report-sha256 cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732 --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --dtype fp16 --weight-mode resident --output-directory "$OUT/tiled-qualification" > "$OUT/tiled-qualification.log" 2>&1
for PHASE in 0 1 2 3; do
if test "$PHASE" = 0 || test "$PHASE" = 3; then LIB="$BASE"; else LIB="$TILED"; fi
$PY -u tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/matched-phase-$PHASE" --new-tokens 32 --repeats 2 --dtype fp16 --weight-mode resident --gpu-index 1 > "$OUT/matched-phase-$PHASE.log" 2>&1
done
for PHASE in 0 1; do
if test "$PHASE" = 0; then LIB="$BASE"; else LIB="$TILED"; fi
$PY -u tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$OUT/story-case.json" --output-directory "$OUT/story-phase-$PHASE" --new-tokens 32 --repeats 2 --dtype fp16 --weight-mode resident --gpu-index 1 > "$OUT/story-phase-$PHASE.log" 2>&1
done
mkdir "$OUT/tiled-profiler"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/tiled-profiler/trace" "$PY" tools/benchmark_qwen_native_resident.py "$MODEL" --library "$TILED" --case "$CASE" --output-directory "$OUT/tiled-profiler/profiled-run" --new-tokens 8 --repeats 1 --dtype fp16 --weight-mode resident --gpu-index 1 --profile-nvtx > "$OUT/tiled-profiler/trace.log" 2>&1
$PY tools/summarize_native_runtime_trace.py "$OUT/tiled-profiler/trace.sqlite" --output "$OUT/tiled-profiler/attribution.json"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
'''.replace('@ARTIFACT_SHA@',identity['artifact_sha256'])
code=f'''import pathlib,subprocess,time,json,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
assert (p/'rows-ablation-exit-code.txt').read_text().strip()=='0'
assert not (p/'final-process.json').exists()
(p/'final-worker.sh').write_text({worker!r})
with (p/'final-worker.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(p/'final-worker.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-tiling-src-v1')
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256={{f:hashlib.sha256((src/f).read_bytes()).hexdigest() for f in ['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/models/qwen_native.py','tools/benchmark_qwen_native_resident.py','tools/summarize_native_runtime_trace.py']}})
(p/'final-process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
