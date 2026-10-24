import hashlib,json,os,subprocess,time
from pathlib import Path
S=Path('/home/data/wangyue/repos/codex-build/Venturi/kv-tier-src-lifetime-v4');O=Path('/home/data/wangyue/artifacts/venturi-kv-tier-lifetime-v4')
P='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python';L=Path(json.loads(Path('/home/data/wangyue/artifacts/venturi-kv-tier-v3/library.json').read_text())['library'])
r=json.loads(L.with_name('build.json').read_text());assert hashlib.sha256(L.read_bytes()).hexdigest()==r['library_sha256']
for f,h in r['source_sha256'].items():assert hashlib.sha256((S/'src/venturi/ops/csrc'/f).read_bytes()).hexdigest()==h
E=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',VENTURI_KV_TIER_LIBRARY=str(L),VENTURI_EXPERIMENT_LIBRARY=str(L),PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{S}/src:{S}/tests:/home/wzu/.local/lib/python3.10/site-packages')
p=subprocess.run([P,'-m','pytest','-xq','tests/test_qwen_kv_tier.py','tests/test_native_experiments.py'],cwd=S,env=E,capture_output=True,timeout=180);(O/'fixtures.log').write_bytes(p.stdout+p.stderr)
assert p.returncode==0,'Lifetime/close-race regression failed'
(O/'source-pins.json').write_text(json.dumps({str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (S/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(x.endswith('.egg-info') for x in p.parts) and p.suffix!='.pyc'},indent=2)+'\n')
(O/'completed.json').write_text(json.dumps(dict(completed=True,library=str(L),library_sha256=r['library_sha256'],native_sources_identical_to_full_27b_v3_benchmark=True,python_change='Lock diagnostics versus owner close; retain session/batch usability after model owner close',unix=time.time()),indent=2)+'\n')
