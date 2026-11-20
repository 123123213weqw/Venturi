from pathlib import Path
import subprocess,json,hashlib
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-native-column-mlp';remote='/home/data/wangyue/repos/codex-build/Venturi/native-column-mlp-src-v3'
subprocess.run(ns['SSH'][:-1]+['mkdir -p '+remote],check=True,timeout=60)
ssh=' '.join(ns['SSH'][:-2]);flags=['--exclude='+v for v in ['.git/','target/','node_modules/','.env','.env.*','.venv*','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']]
r=subprocess.run(['rsync','-az',*flags,'-e',ssh,'src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/'],capture_output=True,text=True,timeout=180);(out/'sync-v3.log').write_text(r.stdout+r.stderr);r.check_returncode()
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};(out/'source-manifest-v3.json').write_text(json.dumps(expected,indent=2)+'\n')
base=Path('/tmp/venturi-native-column-mlp-fixtures-v1-worker.py').read_text();head=base[:base.index('try:\n expected=')].replace('native-column-mlp-src-v1','native-column-mlp-src-v3').replace('venturi-native-column-mlp-fixtures-v1','venturi-native-column-mlp-budget-v4')
body=r'''try:
 expected=EXPECTED
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert actual==expected,'Full source manifest mismatch before use'
 (out/'pre-execution-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 gate=root/'artifacts/venturi-native-column-mlp-fixtures-v1';deadline=time.monotonic()+1800
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];p=pathlib.Path(f'/proc/{pid}/cmdline');assert p.exists() and str(gate/'worker.py').encode() in p.read_bytes(),'Predecessor missing';assert time.monotonic()<deadline,'Bounded predecessor observation elapsed';time.sleep(10)
 assert (gate/'exit-code.txt').read_text().strip()=='0'
 library=(gate/'library.txt').read_text().strip();build=json.loads(pathlib.Path(library).with_name('build.json').read_text());assert len(build['source_sha256'])==10
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'library.txt').write_text(library+chr(10))
 (out/'reused-build-and-fixtures.json').write_text(json.dumps(dict(path=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),source_sha256=build['source_sha256'],prior_full_tests_sha256=hashlib.sha256((gate/'full-tests.log').read_bytes()).hexdigest(),scope='Exact ten native dependencies; old full native suite reused. Expanded new-profile transaction/thread/cohort tests run fresh below.'),indent=2)+chr(10))
 env['VENTURI_NATIVE_COLUMN_MLP_LIBRARY']=library
 parent=root/'artifacts/venturi-native-column-mlp-model-v2';deadline=time.monotonic()+1800
 while not (parent/'exit-code.txt').exists():
  pid=json.loads((parent/'process.json').read_text())['pid'];proc=pathlib.Path(f'/proc/{pid}/cmdline');assert proc.exists() and str(parent/'worker.py').encode() in proc.read_bytes();assert time.monotonic()<deadline;time.sleep(10)
 assert (parent/'exit-code.txt').read_text().strip()=='0'
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json'
 prior=root/'artifacts/venturi-shared-activation-model-v2/qualification/report.json';fresh=parent/'qualification/report.json'
 run([py,'-u','tools/qualify_native_column_mlp.py',str(target),'--library',library,'--cases',str(cases),'--prior-teacher-report',str(prior),'--prior-teacher-report-sha256','073e59da90afa6dc5822c1dee0c2fa9377b33bec107eb0ab710c39edbacf8529','--projection-rows','256','--reuse-qualified-report',str(fresh),'--reuse-qualified-report-sha256',hashlib.sha256(fresh.read_bytes()).hexdigest(),'--output-directory',str(out/'qualification')],'qualification.log',timeout=900)
 assert json.loads((out/'qualification/report.json').read_text())['same_profile_row_budget_bytes_equal']
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,model_qualified=True,operator_promoted=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-native-column-mlp-budget-v4',head+body.replace(' expected=EXPECTED',' expected='+repr(expected)))
