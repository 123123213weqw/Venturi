from pathlib import Path
import subprocess,json,hashlib
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-native-column-mlp';remote='/home/data/wangyue/repos/codex-build/Venturi/native-column-mlp-src-v5'
subprocess.run(ns['SSH'][:-1]+['mkdir -p '+remote],check=True,timeout=60)
ssh=' '.join(ns['SSH'][:-2]);flags=['--exclude='+v for v in ['.git/','target/','node_modules/','.env','.env.*','.venv*','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']]
r=subprocess.run(['rsync','-az',*flags,'-e',ssh,'src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/'],capture_output=True,text=True,timeout=180);(out/'sync-v5.log').write_text(r.stdout+r.stderr);r.check_returncode()
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};(out/'source-manifest-v5.json').write_text(json.dumps(expected,indent=2)+'\n')
base=Path('/tmp/venturi-native-column-mlp-model-v2-worker.py').read_text();head=base[:base.index('try:\n expected=')].replace('native-column-mlp-src-v2','native-column-mlp-src-v5').replace('venturi-native-column-mlp-model-v2','venturi-native-column-mlp-cli-v8')
body=r'''try:
 expected=EXPECTED
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert actual==expected
 (out/'pre-execution-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 gate=root/'artifacts/venturi-native-column-mlp-profile-v7';deadline=time.monotonic()+3600
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];proc=pathlib.Path(f'/proc/{pid}/cmdline');assert proc.exists() and str(gate/'worker.py').encode() in proc.read_bytes();assert time.monotonic()<deadline;time.sleep(10)
 assert (gate/'exit-code.txt').read_text().strip()=='0'
 library=(gate/'library.txt').read_text().strip();build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'library.txt').write_text(library+chr(10))
 run([py,'-m','pytest','-q','tests/test_qwen_native_cli.py','--tb=short'],'cli-tests.log',timeout=120)
 case=json.loads((root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json').read_text())
 cmd=[py,'-m','venturi','qwen-native','generate',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'),'--blocks-library',library,'--weight-mode','resident','--dtype','fp16','--device','cuda:0','--tensor-parallel-device','cuda:1','--tensor-parallel-transport','host','--projection-profile','q8-residual-column-mlp','--max-tokens','32','--capacity','256','--token-ids',','.join(map(str,case['prompt_token_ids'])),'--max-new-tokens',str(len(case['expected_generated_ids'])),'--eos-token-ids','','--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--mtp-draft-tokens','2','--mtp-recovery','prefix']
 run(cmd,'actual-cli.json',timeout=300)
 result=json.loads((out/'actual-cli.json').read_text());assert result['generated_ids']==case['expected_generated_ids']
 assert result['projection_profile']=='q8-residual-column-mlp' and result['native_memory']['column_mlp_enabled']
 assert result['native_memory']['partial_and_host_transport_dtype']=='float32'
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),actual_public_cli_fixed_tokens_match=True,ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-native-column-mlp-cli-v8',head+body.replace(' expected=EXPECTED',' expected='+repr(expected)))
