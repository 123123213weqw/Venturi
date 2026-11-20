import subprocess,pathlib,json,hashlib
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
sha=json.loads((root/'benchmarks/v100-native-tp/v100/full/qualification/report.json').read_text())['artifact_sha256']
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-pp-full-qualification-v1
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-pp-execution-src-v1
export CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VENTURI_PP_TEST=1 VENTURI_TP_TEST=1 VENTURI_EXPECT_P2P_FAILURE=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/v100-pp-validation-v2/library.txt)
export VENTURI_SM70_LIBRARY="$LIB"
MODEL=/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
$PY -u tools/qualify_qwen_native.py "$MODEL" --expected-sha256 @SHA@ --blocks-library "$LIB" --previous-directory /home/wzu/codex-build/Venturi/greedy-validation-v3/full-model-run1 --previous-report-sha256 cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732 --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --dtype fp16 --weight-mode resident --pipeline-device cuda:1 --pipeline-split-layer 32 --output-directory "$OUT/qualification" > "$OUT/qualification.log" 2>&1
$PY -m pytest -q --tb=short > "$OUT/tests.log" 2>&1
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/gpu-final.txt"
'''.replace('@SHA@',sha)
files=['src/venturi/ops/csrc/blocks.cu','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/ops/csrc/qwen_pipeline.cuh','src/venturi/models/qwen_native.py','src/venturi/models/qwen_pipeline.py','tools/qualify_qwen_native.py','tools/benchmark_qwen_native_parallel.py']
expected={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
code=f'''import pathlib,subprocess,time,json,hashlib
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-full-qualification-v1');p.mkdir()
assert pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-validation-v2/exit-code.txt').read_text().strip()=='0'
src=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-execution-src-v1');expected={expected!r}
assert all(hashlib.sha256((src/f).read_bytes()).hexdigest()==h for f,h in expected.items())
(p/'worker.sh').write_text({worker!r})
with (p/'worker.log').open('xb') as log: proc=subprocess.Popen(['bash',str(p/'worker.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time(),source_sha256=expected)
(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
