import pathlib,hashlib,json,shutil,datetime,subprocess
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-cuda-occupancy'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'evidence-sha256.json').exists(),'Already frozen: do not mutate'
assert json.loads((out/'completion-check.json').read_text())['all_terminal_and_report_checks_passed']
prior={}
for cap in sorted((root/'benchmarks').iterdir()):
 if cap==out or not cap.is_dir():continue
 record={}
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  p=cap/name
  if p.exists():
   mapping=json.loads(p.read_text());mapping=mapping.get('files',mapping)
   assert isinstance(mapping,dict)
   checked=[];unavailable=[]
   for f,h in mapping.items():
    target=base/f
    if not target.exists() and cap.name=='decoder-layer':target=base/pathlib.Path(f).name
    if target.is_file():
     assert sha(target)==h,str(target);checked.append(f)
    else:unavailable.append(f)
   if unavailable:assert cap.name=='decoder-layer' and len(checked)==4,'Unexpected missing frozen source'
   record[name]=dict(verified=len(checked),legacy_unavailable_files=unavailable)
 if record:prior[cap.name]=record
native=json.loads((out/'v100/token-aware/candidate/build.json').read_text())['source_sha256']
assert all(sha(root/'src/venturi/ops/csrc'/p)==h for p,h in native.items())
proof=json.loads((out/'v100/final-check/source-and-library.json').read_text())
assert proof['source_sha256']==native and proof['exact_executed_native_source_match']
pre=json.loads((out/'v100/token-aware/prerequisites.json').read_text())
assert pre['previous_qualification_report_sha256']==sha(out/'v100/token-aware-initial/qualification/report.json')
for d in ['variants','token-aware','diagnostics','final-check']:
 assert (out/f'v100/{d}/exit-code.txt').read_text().strip()=='0'
for k in ['all_fixed_cpu_logit_gates_passed','all_native_logit_hidden_state_bytes_match_previous','native_full_runtime_case_passed']:
 assert json.loads((out/'v100/token-aware/qualification/report.json').read_text())[k]
assert (out/'v100/token-aware-initial/exit-code.txt').read_text().strip()=='1'
assert '839 passed, 384 skipped' in (out/'v100/token-aware/tests.log').read_text()
assert '80 passed, 69 skipped' in (out/'v100/final-check/tests.log').read_text()
assert '729 passed, 494 skipped' in pathlib.Path('/tmp/venturi-occupancy-local-final-tests.log').read_text()
assert 'RmProfilingAdminOnly: 1' in (out/'v100/diagnostics/policy.txt').read_text()
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for f in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 p=root/f
 if p.exists():files.append(p)
files += [out/'summary.md',out/'summarize.py']
manifest={}
for p in sorted(set(files)):
 relative=p.relative_to(root);dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(relative)]=sha(dest)
launch=out/'launchers';launch.mkdir()
for p in pathlib.Path('/tmp').glob('venturi-*occupancy*.py'):
 shutil.copy2(p,launch/p.name)
shutil.copy2('/tmp/venturi-occupancy-local-tests.log',out/'local-tests-before-promotion.log')
shutil.copy2('/tmp/venturi-occupancy-local-final-tests.log',out/'local-tests.log')
r=subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools','benchmarks/v100-cuda-occupancy/summarize.py'],cwd=root,text=True,capture_output=True,check=True)
(out/'ruff.log').write_text(r.stdout+r.stderr)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
(out/'evidence-check.json').write_text(json.dumps(dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),exact_executed_native_source_match=True,prior_capsules_unchanged=prior,source_code_change='SM70-only token-aware compiler launch bounds, unchanged 256-thread actual geometry and numerical operations',hardware_counter_policy_unchanged=True,achieved_occupancy=None,original_source_sync_failure_preserved=True,all_required_final_jobs_completed=True,local_rust_compilation=False,no_commit_or_push=True),indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
assert all(sha(out/f)==h for f,h in evidence.items())
assert all(sha(out/'source'/f)==h for f,h in manifest.items())
print(json.dumps(dict(source_files=len(manifest),evidence_files=len(evidence),prior_capsules=len(prior))))
