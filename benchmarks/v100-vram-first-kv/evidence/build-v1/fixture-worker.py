import json,os,subprocess,hashlib,time
from pathlib import Path
S=Path('/home/data/wangyue/repos/codex-build/Venturi/kv-tier-src-v1');O=Path('/home/data/wangyue/artifacts/venturi-kv-tier-v2');O.mkdir(exist_ok=False)
P='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
E=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{S}/src:{S}/tests:/home/wzu/.local/lib/python3.10/site-packages')
def run(n,args,timeout):
 p=subprocess.run(args,cwd=S,env=E,capture_output=True,timeout=timeout);(O/(n+'.log')).write_bytes(p.stdout+p.stderr);assert p.returncode==0,n
run('build',[P,'-m','venturi.ops.build','--component','blocks','--architecture','sm70','--q8-row-pair','--q8-swiglu-quant','--q8-rms-quant','--nvcc','/usr/local/cuda/bin/nvcc','--output-dir',str(O/'build')],240)
lib=(O/'build.log').read_text().strip().splitlines()[-1];(O/'library.json').write_text(json.dumps(dict(library=lib)))
E.update(VENTURI_KV_TIER_LIBRARY=lib,VENTURI_EXPERIMENT_LIBRARY=lib)
run('vmm-unit-build',['/usr/local/cuda/bin/nvcc','-O2','-std=c++20','-I','src/venturi/ops/csrc','tests/csrc/kv_tier_probe.cu','-lcuda','-o',str(O/'kv_tier_probe')],45)
run('vmm-unit',[str(O/'kv_tier_probe')],30)
run('fixtures',[P,'-m','pytest','-xq','tests/test_qwen_kv_tier.py','tests/test_native_experiments.py'],200)
(O/'executed-source-pins.json').write_text(json.dumps({str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (S/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(x.endswith('.egg-info') for x in p.parts) and p.suffix!='.pyc'},indent=2))
(O/'completed.json').write_text(json.dumps(dict(completed=True,library=lib,unix=time.time())))
