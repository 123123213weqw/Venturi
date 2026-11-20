import subprocess,pathlib,hashlib,json
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
ids=','.join(map(str,json.loads((root/'benchmarks/qwen-streaming/case.json').read_text())['prompt_token_ids']))
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-diagnostics-v2
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-final-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_PP_TEST=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-pp-final-v1/library.txt)
export VENTURI_SM70_LIBRARY="$LIB"
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
CASE=/home/wzu/codex-build/Venturi/v100-pp-final-v1/case.json
for MODE in tp pp; do
for COUNT in 1 8; do
D="$OUT/$MODE-c$COUNT";mkdir "$D"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$D/trace" "$PY" tools/benchmark_qwen_native_parallel.py "$MODEL" --library "$LIB" --case "$CASE" --mode "$MODE" --output-directory "$D/profiled-run" --concurrency "$COUNT" --repeats 1 --new-tokens 8 --split-layer 32 --profile-nvtx > "$D/trace.log" 2>&1
$PY tools/summarize_native_parallel_trace.py "$D/trace.sqlite" --output "$D/cohort-attribution.json"
done
done
$PY -m venturi qwen-native generate "$MODEL" --blocks-library "$LIB" --weight-mode resident --dtype fp16 --token-ids @IDS@ --max-new-tokens 2 --max-tokens 26 --eos-token-ids 248046,248044 --pipeline-device cuda:1 --pipeline-split-layer 32 > "$OUT/cli-smoke.json" 2> "$OUT/cli-smoke.log"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/hardware-counter-policy.txt"
nvidia-smi topo -m > "$OUT/topology.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
'''.replace('@IDS@',ids)
files=['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/ops/csrc/qwen_pipeline.cuh','src/venturi/models/qwen_native.py','src/venturi/models/qwen_pipeline.py','tools/benchmark_qwen_native_parallel.py','tools/summarize_native_parallel_trace.py','tests/test_native_parallel_trace.py','src/venturi/qwen_native_cli.py']
expected={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
code=f'''import pathlib,subprocess,time,json,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-diagnostics-v2');p.mkdir()
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-final-src-v1');expected={expected!r}
assert all(hashlib.sha256((src/f).read_bytes()).hexdigest()==h for f,h in expected.items())
(p/'worker.sh').write_text({worker!r})
waiter=\"\"\"import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-final-v1/exit-code.txt')
for _ in range(240):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Unprofiled benchmarks did not finish within 40 minutes; no profiling started')
assert q.read_text().strip()=='0','Unprofiled benchmark failed; no profiler started'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-pp-diagnostics-v2/worker.sh']))
\"\"\"
(p/'waiter.py').write_text(waiter)
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'waiter.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256=expected,scheduled_after='Unprofiled ABBA terminal success')
(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(pid=proc.pid)))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
