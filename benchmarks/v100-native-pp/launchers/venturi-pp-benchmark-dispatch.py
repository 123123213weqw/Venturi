import pathlib,json,subprocess,hashlib
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
old=root/'benchmarks/v100-native-tp/v100/full/story-phase-0/report.json';r=json.loads(old.read_text());case=dict(r['case']);case['expected_generated_ids']=r['warmup_ids'];case['expected_ids_source']='Pinned original single-native Q8/FP16 ABI-13 report';case['expected_report_sha256']=hashlib.sha256(old.read_bytes()).hexdigest()
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-comparison-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-execution-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-pp-validation-v2/library.txt)
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
for PHASE in 0 1 2 3; do
if test "$PHASE" = 0 || test "$PHASE" = 3; then MODE=tp; else MODE=pp; fi
$PY -u tools/benchmark_qwen_native_parallel.py "$MODEL" --library "$LIB" --case "$OUT/case.json" --mode "$MODE" --output-directory "$OUT/phase-$PHASE" --concurrency 1,2,4,8 --repeats 1 --new-tokens 32 --split-layer 32 > "$OUT/phase-$PHASE.log" 2>&1
done
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
'''
files=['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/ops/csrc/qwen_pipeline.cuh','src/venturi/models/qwen_native.py','src/venturi/models/qwen_pipeline.py','tools/benchmark_qwen_native_parallel.py']
expected={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
code=f'''import pathlib,subprocess,time,json,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-comparison-v1');p.mkdir()
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-execution-src-v1');expected={expected!r}
assert all(hashlib.sha256((src/f).read_bytes()).hexdigest()==h for f,h in expected.items())
(p/'case.json').write_text({json.dumps(case,indent=2)!r});(p/'worker.sh').write_text({worker!r})
waiter=\"\"\"import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-full-qualification-v1/exit-code.txt')
for _ in range(180):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Qualification did not finish within 30 minutes; timing not started')
assert q.read_text().strip()=='0','Correctness gate failed: timing not started'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-pp-comparison-v1/worker.sh']))
\"\"\"
(p/'waiter.py').write_text(waiter)
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'waiter.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256=expected,scheduled_after='Full trained-model and full-suite terminal success')
(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(pid=proc.pid)))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
