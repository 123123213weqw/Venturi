import pathlib,hashlib,json,shutil,datetime,subprocess
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-native-mtp';raw=out/'v100'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
assert not (out/'evidence-sha256.json').exists(),'Already frozen; never mutate'
for d in ['validation-v1','export-v1','qualification-v3','final-v4','diagnostics-v1']:
 assert (raw/d/'exit-code.txt').read_text().strip()=='0',d
 assert read(raw/d/'completed.json')['completed'],d
assert (raw/'qualification-v2/exit-code.txt').read_text().strip()=='1'
assert 'KeyError' in (raw/'qualification-v2/mtp-tp.log').read_text()
assert 'centered_cosine' in (raw/'qualification-v2/mtp-tp.log').read_text()
assert not read(raw/'qualification-v2/mtp-tp/report.json')['completed']
qual=read(raw/'qualification-v2/target-qualification/report.json')
for k in ['all_fixed_cpu_logit_gates_passed','all_native_logit_hidden_state_bytes_match_previous','native_full_runtime_case_passed']:assert qual[k]
pre=read(raw/'qualification-v3/prerequisites.json')
assert pre['target_qualification_report_sha256']==sha(raw/'qualification-v2/target-qualification/report.json')
proof=read(raw/'diagnostics-v1/executed-source.json');build=read(raw/'build-final/build.json')
assert proof['library_sha256']==build['library_sha256']==read(raw/'diagnostics-v1/completed.json')['library_sha256']
assert not proof['owned_task_processes']
assert all(sha(root/f)==h for f,h in proof['source_sha256'].items())
assert all(sha(root/'src/venturi/ops/csrc'/f)==h for f,h in build['source_sha256'].items())
for mode in ['tp','pp']:
 q=read(raw/f'final-v4/qualification-{mode}/report.json');assert q['completed'] and len(q['block_reference'])==2 and len(q['cases'])==4
 assert all(b['passed'] for b in q['block_reference'])
 assert all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and not c['mtp_stats']['forced_proposals'] for c in q['cases'])
 r=read(raw/f'final-v4/comparison-{mode}/report.json');assert r['completed'] and not r['profiled'] and len(r['trials'])==16
 assert r['library_sha256']==proof['library_sha256']
 assert all(c['cohort']['emitted_tokens']==32*c['request_count'] and all(s['forced_proposals']==0 for s in c['mtp_stats']) for c in r['trials'])
for label in ['tp-c1','pp-c8']:
 r=read(raw/f'diagnostics-v1/{label}/profiled-run/report.json');assert r['completed'] and r['profiled'] and r['new_tokens']==8
 assert r['library_sha256']==proof['library_sha256']
assert '864 passed, 403 skipped' in (raw/'final-v4/tests.log').read_text()
assert '92 passed, 88 skipped' in (raw/'final-v4/fixture-tests.log').read_text()
assert '735 passed, 532 skipped' in pathlib.Path('/tmp/venturi-mtp-local-final-tests.log').read_text()
assert 'RmProfilingAdminOnly: 1' in (raw/'diagnostics-v1/policy.txt').read_text()
assert read(raw/'diagnostics-v1/cli-smoke.json')['speculative_draft_tokens']==1
prior={}
for cap in sorted((root/'benchmarks').iterdir()):
 if cap==out or not cap.is_dir():continue
 record={}
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  path=cap/name
  if not path.exists():continue
  mapping=read(path);mapping=mapping.get('files',mapping);assert isinstance(mapping,dict)
  checked=[];unavailable=[]
  for f,h in mapping.items():
   target=base/f
   if not target.exists() and cap.name=='decoder-layer':target=base/pathlib.Path(f).name
   if target.is_file():assert sha(target)==h,str(target);checked.append(f)
   else:unavailable.append(f)
  if unavailable:assert cap.name=='decoder-layer' and len(checked)==4,'Unexpected missing prior evidence'
  record[name]=dict(verified=len(checked),legacy_unavailable_files=unavailable)
 if record:prior[cap.name]=record
(out/'completion-check.json').write_text(json.dumps(dict(all_terminal_and_report_checks_passed=True,executed_source_files_verified=len(proof['source_sha256']),expected_failed_attempt_preserved=True,exact_library_and_native_source_match=True,no_owned_gpu_jobs_remaining=True,achieved_hardware_counters_available=False,prior_capsules_unchanged=prior),indent=2)+'\n')
launch=out/'launchers';launch.mkdir(exist_ok=True)
for p in pathlib.Path('/tmp').glob('venturi-mtp-*.py'):
 shutil.copy2(p,launch/p.name)
shutil.copy2('/tmp/venturi-freeze-mtp.py',launch/'freeze.py')
shutil.copy2('/tmp/venturi-mtp-cases.json',out/'trained-cases.json')
shutil.copy2('/tmp/venturi-mtp-local-final-tests.log',out/'local-tests.log')
for p in pathlib.Path('/tmp').glob('venturi-mtp-*tests.log'):
 if p.name!='venturi-mtp-local-final-tests.log':shutil.copy2(p,out/p.name)
r=subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools','benchmarks/v100-native-mtp/summarize.py'],cwd=root,text=True,capture_output=True,check=True)
(out/'ruff.log').write_text(r.stdout+r.stderr)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for f in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 p=root/f
 if p.exists():files.append(p)
files += [out/'summary.md',out/'summarize.py']
manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,check=True)
(out/'evidence-check.json').write_text(json.dumps(dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),qualified_native_source_match=True,qualified_python_source_match=True,source_code_change='Opt-in original-source one-proposal MTP, transactional GDN verification/rollback and exact SM70 T2 tile dispatch',all_required_final_jobs_completed=True,prior_capsules_unchanged=prior,hardware_counter_policy_unchanged=True,achieved_occupancy=None,known_failed_reporting_attempt_preserved=True,mtp_default=False,longer_chains_sampling_continuation_batching_paging_graphs_unqualified=True,external_engine_win_claim=False,no_commit_or_push=True,local_rust_compilation=False),indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
assert all(sha(out/f)==h for f,h in evidence.items())
assert all(sha(out/'source'/f)==h for f,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules=len(prior))))
