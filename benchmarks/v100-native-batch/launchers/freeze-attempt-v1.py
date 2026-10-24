import pathlib,hashlib,json,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-native-batch';raw=out/'v100'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
assert not (out/'evidence-sha256.json').exists(),'Already frozen; never modify'
expected={'initial-fixtures-v1':1,'fixtures-v2':0,'campaign-v3':1,'campaign-v4':1,'final-v5':0,'diagnostics-v6':0}
for name,code in expected.items():assert int((raw/name/'exit-code.txt').read_text())==code,name
assert '14 passed, 13 skipped' in (raw/'fixtures-v2/batch-tests.log').read_text()
assert '50 passed, 46 skipped' in (raw/'fixtures-v2/scalar-regression-tests.log').read_text()
assert '19 passed' in (raw/'fixtures-v2/admission-tests.log').read_text()
assert '856 passed, 458 skipped' in (raw/'final-v5/tests.log').read_text()
assert '756 passed, 558 skipped' in (out/'local-tests.log').read_text()
assert 'AssertionError' in (raw/'initial-fixtures-v1/batch-tests.log').read_text()
assert 'Admission split a fixed cohort' in (raw/'campaign-v3/failure.txt').read_text()
assert 'RemoteDisconnected' in (raw/'campaign-v4/external-layer-mtp1/benchmark.log').read_text()
assert 'RemoteDisconnected' in (raw/'diagnostics-v6/external-layer-mtp1-fresh-c8/benchmark.log').read_text()
qual=read(raw/'campaign-v4/qualification/report.json');assert qual['completed'] and len(qual['requests'])==8
assert all(r['state_bytes_exact'] and r['generated_ids']==r['scalar_ids'] for r in qual['requests'])
assert qual['reordered_unequal_position_continuation']['logits_bytes_exact']
assert qual['reordered_unequal_position_continuation']['all_slot_states_bytes_exact']
proof=read(raw/'final-v5/executed-source.json');build=read(raw/'initial-fixtures-v1/build/a4d0656f0f653e0a/build.json')
assert qual['library_sha256']==proof['library_sha256']==build['library_sha256']
assert proof['exact_source_unchanged']
for path,h in proof['source_sha256'].items():assert sha(root/path)==h,path
for path,h in build['source_sha256'].items():assert sha(root/'src/venturi/ops/csrc'/path)==h,path
for label in ['promotion-scalar','promotion-batch','external-tensor-off']:
 r=read(raw/'campaign-v4'/label/'timed/report.json');assert r['completed'] and len(r['waves'])==12
 assert {w['request_count'] for w in r['waves']}=={1,2,4,8}
 for w in r['waves']:
  assert w['emitted_tokens']==32*w['request_count']
  for q in w['requests']:
   t=q['response'];assert t['tokens']==r['case']['expected_generated_ids'] and t['tokens_evaluated']==23
   assert t.get('timings',{}).get('prompt_n',23)==23
   if label=='promotion-batch':assert t['batch_request_count']==w['request_count']
  if label=='promotion-batch':assert len({q['response']['batch_cohort_id'] for q in w['requests']})==1
assert not read(raw/'campaign-v4/external-layer-mtp1/timed/report.json')['completed']
assert read(raw/'diagnostics-v6/campaign-report.json')['completed']
for name,ranges in [('scalar',8),('batch',1)]:
 m=read(raw/f'diagnostics-v6/profile-{name}-metadata.json');assert m['profiled'] and m['logical_requests']==8 and m['native_call_ranges']==ranges
 s=read(raw/f'diagnostics-v6/profile-{name}-summary.json');assert len(s['cohorts'])==1 and s['cohorts'][0]['requests']==ranges
occupancy=read(raw/'diagnostics-v6/theoretical-occupancy.json');assert occupancy['library_sha256']==proof['library_sha256'] and occupancy['achieved_occupancy'] is None and not occupancy['hardware_counter_collection']
audit=read(out/'remote-terminal-audit.json');assert not audit['owned_inference_processes'];assert all(w['state'] in ['gone','Z'] for w in audit['workers'].values());assert 'RmProfilingAdminOnly: 1' in audit['counter_policy']
prior={}
for cap in sorted((root/'benchmarks').iterdir()):
 if cap==out or not cap.is_dir():continue
 rec={}
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  path=cap/name
  if not path.exists():continue
  mapping=read(path);mapping=mapping.get('files',mapping);checked=0;missing=[]
  for f,h in mapping.items():
   target=base/f
   if not target.exists() and cap.name=='decoder-layer':target=base/pathlib.Path(f).name
   if target.is_file():assert sha(target)==h,str(target);checked+=1
   else:missing.append(f)
  if missing:assert cap.name=='decoder-layer' and checked==4
  rec[name]=dict(verified=checked,legacy_unavailable_files=missing)
 if rec:prior[cap.name]=rec
assert len(prior)==17
shutil.copy2('/tmp/venturi-batch-freeze.py',out/'launchers/freeze.py')
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools','benchmarks/v100-native-batch/summarize.py'],cwd=root,stdout=(out/'ruff.log').open('w'),check=True)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 p=root/name
 if p.exists():files.append(p)
files += [out/'summary.md',out/'summarize.py']
manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,check=True)
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_batch_correctness_qualified=True,native_3_repeat_1_2_4_8_matrix_complete=True,external_non_speculative_3_repeat_matrix_complete=True,external_mtp_matrix_complete=False,failed_attempts_preserved=True,all_owned_workers_terminal=True,no_owned_inference_processes_remaining=True,original_gpu_footprint_restored=True,library_sha256=proof['library_sha256'],executed_source_files_verified=len(proof['source_sha256']),source_files=len(manifest),hardware_counter_policy_unchanged=True,achieved_occupancy=None,profile_scope='23 prompt tokens, 8 outputs, 8 logical requests; separate from 32-output unprofiled timing',default_batch_admission_ms=5,benchmark_batch_admission_ms=25,continuous_batching_paging_pp_mtp_batching_sampling_graphs_supported=False,external_speed_goal_complete=False,no_commit_or_push=True,local_rust_compilation=False,prior_capsules_unchanged=prior)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
assert all(sha(out/f)==h for f,h in evidence.items());assert all(sha(out/'source'/f)==h for f,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),goal_complete=False)))
