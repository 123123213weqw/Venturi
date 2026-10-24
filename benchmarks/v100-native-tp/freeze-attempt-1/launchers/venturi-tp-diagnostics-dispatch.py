import subprocess,json,pathlib,hashlib
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
ids=','.join(map(str,json.load((root/'benchmarks/qwen-streaming/case.json').open())['prompt_token_ids']))
probe='''import torch,json
r=dict(cuda_visible_devices="1,0",devices=[torch.cuda.get_device_name(i) for i in range(2)],directions=[])
for src,dst in [(0,1),(1,0)]:
 x=(torch.arange(1024,device=f"cuda:{src}",dtype=torch.float32)*3+17)
 y=torch.full((1024,),-999.,device=f"cuda:{dst}")
 y.copy_(x);torch.cuda.synchronize(src);torch.cuda.synchronize(dst)
 actual=y.cpu();expected=x.cpu()
 row=dict(source_logical_device=src,destination_logical_device=dst,advertised_peer_access=torch.cuda.can_device_access_peer(src,dst),direct_copy_bytes_equal=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),mismatched_values=int((actual!=expected).sum()),first_expected=expected[:8].tolist(),first_actual=actual[:8].tolist())
 y.copy_(expected,non_blocking=False);torch.cuda.synchronize(dst)
 row["explicit_host_copy_bytes_equal"]=torch.equal(y.cpu().view(torch.uint8),expected.view(torch.uint8));r["directions"].append(row)
print(json.dumps(r,indent=2))
'''
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tp-tools-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-tp-validation-v2/library.txt)
export VENTURI_SM70_LIBRARY="$LIB"
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
CASE=/home/wzu/codex-build/Venturi/streaming-validation/case.json
$PY -m pytest -q --tb=short > "$OUT/tests.log" 2>&1
$PY "$OUT/peer-probe.py" > "$OUT/peer-probe.json" 2> "$OUT/peer-probe.log"
for MODE in single parallel; do
mkdir "$OUT/$MODE-profiler"
if test "$MODE" = parallel; then
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/$MODE-profiler/trace" "$PY" tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/$MODE-profiler/profiled-run" --new-tokens 8 --repeats 1 --dtype fp16 --weight-mode resident --gpu-index 1 --profile-nvtx --tensor-parallel-device cuda:1 --tensor-parallel-transport host --peer-gpu-index 0 > "$OUT/$MODE-profiler/trace.log" 2>&1
else
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --stats=true --output="$OUT/$MODE-profiler/trace" "$PY" tools/benchmark_qwen_native_resident.py "$MODEL" --library "$LIB" --case "$CASE" --output-directory "$OUT/$MODE-profiler/profiled-run" --new-tokens 8 --repeats 1 --dtype fp16 --weight-mode resident --gpu-index 1 --profile-nvtx > "$OUT/$MODE-profiler/trace.log" 2>&1
fi
$PY tools/summarize_native_runtime_trace.py "$OUT/$MODE-profiler/trace.sqlite" --output "$OUT/$MODE-profiler/attribution.json"
done
$PY -m venturi qwen-native generate "$MODEL" --blocks-library "$LIB" --weight-mode resident --dtype fp16 --token-ids @IDS@ --max-new-tokens 2 --max-tokens 26 --eos-token-ids 248046,248044 --tensor-parallel-device cuda:1 --tensor-parallel-transport host > "$OUT/cli-smoke.json" 2> "$OUT/cli-smoke.log"
nvidia-smi topo -m > "$OUT/topology.txt"
grep RmProfilingAdminOnly /proc/driver/nvidia/params > "$OUT/hardware-counter-policy.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
'''.replace('@IDS@',ids)
code=f'''import pathlib,subprocess,time,json,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v1');p.mkdir(exist_ok=True);assert not (p/'process.json').exists()
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-tp-tools-v1')
expected={ {f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/models/qwen_native.py','tools/benchmark_qwen_native_resident.py','tools/summarize_native_runtime_trace.py','tests/test_ada_build.py','tests/test_qwen_native_cli.py','tests/test_native_runtime_trace.py']}!r}
assert all(hashlib.sha256((src/f).read_bytes()).hexdigest()==h for f,h in expected.items())
(p/'peer-probe.py').write_text({probe!r});(p/'worker.sh').write_text({worker!r})
waiter=\"\"\"import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-tp-full-v1/exit-code.txt')
for _ in range(120):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Benchmark did not finish within bounded wait; no diagnostics started')
assert q.read_text().strip()=='0','Full qualification/benchmark failed; diagnostics not started'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v1/worker.sh']))
\"\"\"
(p/'waiter.py').write_text(waiter)
with (p/'worker.log').open('xb') as log:
 proc=subprocess.Popen(['python3',str(p/'waiter.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256=expected,scheduled_after='v100-tp-full-v1 terminal success')
(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
