from pathlib import Path
import subprocess,shlex,json,hashlib,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
root=Path.cwd();out=root/'benchmarks/v100-source-scope';remote='/home/data/wangyue/repos/codex-build/Venturi/source-scope-src-v3'
r=subprocess.run(ns['SSH'],input=f"import pathlib\np=pathlib.Path({remote!r});p.mkdir()\n",capture_output=True,text=True,timeout=60);r.check_returncode()
cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
for pattern in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',pattern]
cmd+=['src','tests','tools','scripts','docs','README.md','pyproject.toml','AGENTS.md','WZU_Server:'+remote+'/']
for attempt in range(2):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (out/'sync-v3.log').open('a') as f:f.write(str(attempt)+'\n'+r.stdout+r.stderr)
 if not r.returncode:break
 if attempt==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};(out/'source-manifest-v3.json').write_text(json.dumps(expected,indent=2)+'\n')
refs={
 'target':('benchmarks/v100-native-column-mlp/v100/model-v2/qualification/report.json','/home/data/wangyue/artifacts/venturi-native-column-mlp-model-v2/qualification/report.json'),
 'draft':('benchmarks/v100-mtp-draft-residual/v100/model-v2/q8-residual-qualification/report.json','/home/data/wangyue/artifacts/venturi-mtp-draft-residual-model-v2/q8-residual-qualification/report.json'),
 'batch':('benchmarks/v100-mtp-draft-residual/v100/model-v2/batch-qualification/report.json','/home/data/wangyue/artifacts/venturi-mtp-draft-residual-model-v2/batch-qualification/report.json')}
pins={name:dict(path=remote,sha256=hashlib.sha256(Path(local).read_bytes()).hexdigest()) for name,(local,remote) in refs.items()};(out/'reused-quality-reference-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
old=Path('/tmp/venturi-source-validation-probe-v1-worker.py').read_text();source=old[:old.index('sampler=None;')].replace('source-validation-src-v1','source-scope-src-v3').replace('venturi-source-validation-probe-v1','venturi-source-scope-model-v3')
source+='try:\n expected='+repr(expected)+'\n refs='+repr(pins)+r'''
 gate=root/'artifacts/venturi-source-scope-fixtures-v2';deadline=time.monotonic()+3600
 while not (gate/'exit-code.txt').exists():
  pid=json.loads((gate/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(gate/'worker.py').encode() in q.read_bytes(),'Prerequisite missing without terminal marker'
  assert time.monotonic()<deadline,'Live prerequisite bounded wait elapsed';time.sleep(3)
 assert (gate/'exit-code.txt').read_text().strip()=='0','Fixture gate failed'
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 library=pathlib.Path((gate/'library.txt').read_text().strip());build=json.loads(library.with_name('build.json').read_text());assert build['resident_source_scope'] and sha(library)==build['library_sha256']
 for rel,h in build['source_sha256'].items():assert sha(src/'src/venturi/ops/csrc'/rel)==h
 (out/'library.txt').write_text(str(library)+chr(10))
 for ref in refs.values():assert sha(pathlib.Path(ref['path']))==ref['sha256']
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json'
 payloads={str(p):identity(p) for p in [target,draft]};(out/'payload-before.json').write_text(json.dumps(payloads,indent=2)+chr(10))
 cmd=[py,'tools/qualify_qwen_execution_policy.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--cases',str(cases),'--expected-source-scope','1','--output-directory',str(out/'qualification')]
 for name,ref in refs.items():cmd+=['--prior-'+name+'-report',ref['path'],'--prior-'+name+'-report-sha256',ref['sha256']]
 run(cmd,'qualification.log',timeout=900)
 run([py,'tools/qualify_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--cases',str(cases),'--prior-target-report',refs['target']['path'],'--prior-target-report-sha256',refs['target']['sha256'],'--projection-profile','q8-residual-column-mlp','--mtp-draft-projection-profile','q8-residual','--output-directory',str(out/'batch-qualification')],'batch-qualification.log',timeout=900)
 prior=json.loads(pathlib.Path(refs['batch']['path']).read_text());candidate=json.loads((out/'batch-qualification/report.json').read_text());assert candidate['completed'] and candidate['selective_reset_untouched_state_exact']
 assert candidate['full_references']==prior['full_references']
 for a,b in zip(candidate['cohorts'],prior['cohorts']):
  for key in ['slots','budgets','generated_ids','target_states','draft_states']:assert a[key]==b[key],key
 case=json.loads((root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json').read_text())
 run([py,'-m','venturi','qwen-native','generate',str(target),'--blocks-library',str(library),'--weight-mode','resident','--dtype','fp16','--device','cuda:0','--tensor-parallel-device','cuda:1','--tensor-parallel-transport','host','--projection-profile','q8-residual-column-mlp','--max-tokens','32','--capacity','256','--token-ids',','.join(map(str,case['prompt_token_ids'])),'--max-new-tokens','32','--eos-token-ids','','--mtp-artifact',str(draft),'--mtp-draft-tokens','2','--mtp-recovery','prefix','--mtp-draft-projection-profile','q8-residual'],'actual-cli.json')
 cli=json.loads((out/'actual-cli.json').read_text());assert cli['generated_ids']==case['expected_generated_ids']
 assert cli['native_memory']['source_validation_policy'].startswith('resident-forward')
 after={str(p):identity(p) for p in [target,draft]};assert after==payloads;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,model_qualified=True,library_sha256=sha(library),independent_numerical_reference_explicitly_reused=True,fresh_trained_exact_scalar_and_cohort_states=True,original_corrected_draft_bytes_equal=True,actual_cli_passed=True,ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-source-scope-model-v3',source)
