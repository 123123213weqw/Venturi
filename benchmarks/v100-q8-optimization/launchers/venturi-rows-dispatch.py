import subprocess,json
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20']
worker='''set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/rows-ablation-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
export VENTURI_Q8_BASELINE_LIBRARY=/home/data/wangyue/repos/codex-build/Venturi/sm70-build/9b826970163c1fb4/libventuri_blocks.so
for ROW in 2 8; do
export VENTURI_SM70_LIBRARY=$($PY -c "import json; print(next(e['library'] for e in json.load(open('$OUT/row-libraries.json')) if e['row_tile']==$ROW))")
$PY -m pytest -q tests/test_q8_tiling.py --tb=short > "$OUT/tiling-tests-t4-r$ROW.log" 2>&1
done
$PY -u tools/benchmark_q8_tiles.py /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --baseline "$VENTURI_Q8_BASELINE_LIBRARY" --candidates "$OUT/row-libraries.json" --output-directory "$OUT/row-ablation" --gpu-index 1 --decode-only > "$OUT/row-ablation.log" 2>&1
'''
code=f'''import pathlib,json,subprocess,time
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
assert (p/'more-build-v2-exit-code.txt').read_text().strip()=='0'
assert (p/'ablation-v2-exit-code.txt').read_text().strip()=='0'
assert not (p/'rows-ablation-process.json').exists()
r=[e for e in json.load((p/'more-tiling-libraries-v2.json').open()) if e['token_tile']==4]
(p/'row-libraries.json').write_text(json.dumps(r,indent=2))
(p/'rows-ablation.sh').write_text({worker!r})
with (p/'rows-ablation-worker.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(p/'rows-ablation.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'rows-ablation-process.json').write_text(json.dumps(r));print(json.dumps(r))
'''
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
