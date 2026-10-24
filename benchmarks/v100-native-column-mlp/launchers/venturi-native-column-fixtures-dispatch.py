from pathlib import Path
import subprocess,json,hashlib
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-native-column-mlp';remote='/home/data/wangyue/repos/codex-build/Venturi/native-column-mlp-src-v1'
subprocess.run(ns['SSH'][:-1]+['mkdir -p '+remote],check=True,timeout=60)
ssh=' '.join(ns['SSH'][:-2]);flags=['--exclude='+v for v in ['.git/','target/','node_modules/','.env','.env.*','.venv*','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']]
r=subprocess.run(['rsync','-az',*flags,'-e',ssh,'src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/'],capture_output=True,text=True,timeout=180);(out/'sync-v1.log').write_text(r.stdout+r.stderr);r.check_returncode()
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};(out/'source-manifest-v1.json').write_text(json.dumps(expected,indent=2)+'\n')
old=Path('/tmp/venturi-column-tp-fixtures-v1-worker.py').read_text();head=old[:old.index('try:\n expected=')].replace('column-tp-src-v1','native-column-mlp-src-v1').replace('venturi-column-tp-fixtures-v1','venturi-native-column-mlp-fixtures-v1')
start=old.index(" baseline=root/");end=old.index(" env.pop('VENTURI_TILE_ORDER_LIBRARY',None);env.pop('VENTURI_T16_LIBRARY',None);env.pop('VENTURI_LAYER_GRAPH_LIBRARY',None)",start)
envs=old[start:end]
body=r'''try:
 expected=EXPECTED
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert actual==expected,'Full source manifest mismatch before build'
 (out/'pre-build-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 buildcmd="from venturi.ops.build import build_blocks_backend;from pathlib import Path;p=build_blocks_backend("+repr(str(out/'build'))+",nvcc='/usr/local/cuda/bin/nvcc',architecture='sm70');Path("+repr(str(out/'library.txt'))+").write_text(str(p)+chr(10))"
 run([py,'-c',buildcmd],'build.log',timeout=600)
 library=(out/'library.txt').read_text().strip();env['VENTURI_NATIVE_COLUMN_MLP_LIBRARY']=library;env['VENTURI_COLUMN_TP_LIBRARY']=library
 run([py,'-m','pytest','-x','-q','tests/test_native_column_mlp.py','--tb=short'],'focused-tests.log',timeout=600)
 ENVS
 env.pop('VENTURI_TILE_ORDER_LIBRARY',None);env.pop('VENTURI_T16_LIBRARY',None);env.pop('VENTURI_LAYER_GRAPH_LIBRARY',None)
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=600)
 run([py,'tools/inspect_cuda_occupancy.py','--library',library,'--expected-sha256',hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),'--output',str(out/'resources.json')],'resources.log',timeout=180)
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-native-column-mlp-fixtures-v1',head+body.replace(' expected=EXPECTED',' expected='+repr(expected)).replace(' ENVS',envs))
