import pathlib,json,hashlib,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-shared-inputs';raw=out/'v100'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'evidence-sha256.json').exists(),'Already frozen'
for name in ['fixtures-v1','qualification-v2','campaign-v3','final-v4']:assert int((raw/name/'exit-code.txt').read_text())==0,name
assert '1036 passed, 440 skipped' in (raw/'final-v4/tests.log').read_text()
assert '760 passed, 716 skipped' in (out/'local-tests.log').read_text()
assert '40 passed' in (raw/'fixtures-v1/profile-tests.log').read_text()
assert '130 passed, 125 skipped' in (raw/'fixtures-v1/default-regression-tests.log').read_text()
qualified=read(raw/'qualification-v2/qualification/report.json');assert qualified['completed'] and not qualified['broad_bf16_corpus_quality_qualified']
for profile,controls in qualified['profiles'].items():
 assert controls['same_profile_bytes_equal'] and controls['unshared']==controls['shared']
 for mode in ['unshared','shared']:
  p=controls[mode];assert p['passed'] and len(p['cases'])==4
  assert sum(len(c['all_logit_bytes']) for c in p['cases'])==98
  assert all(c['native_generate_without_torch_math'] and c['diagnostic_generate_state_bytes_equal'] for c in p['cases'])
  assert len(p['batch']['private_cache'])==8 and len(p['reset_continuation']['private_cache'])==8
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
assert qualified['prior_report_sha256']==sha(root/'benchmarks/v100-native-residual/v100/qualification-v2/qualification/report.json')
mtp=read(raw/'campaign-v3/mtp-qualification/report.json');assert mtp['completed'] and mtp['projection_profile']=='q8-residual' and mtp['mtp_draft_projection_profile']=='fp16'
assert len(mtp['cases'])==4 and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] for c in mtp['cases'])
campaign=read(raw/'campaign-v3/campaign-report.json');assert campaign['completed'] and campaign['promoted'] and len(campaign['experiments'])==14
for e in campaign['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v3'/e['label']/'timed/report.json');assert r['completed']
 if e['label'].startswith('repeat'):
  counts=[1,2,4,8] if e['label'] in ['repeat-unshared-q8-residual','repeat-shared-q8-residual','repeat-llama-tensor-off'] else ([1,8] if e['label'].endswith('fp16') else [1])
  assert [s['request_count'] for s in e['summary']]==counts and all(s['trials']==3 for s in e['summary'])
  assert len(r['waves'])==len(counts)*3
 if e['engine']=='venturi':
  wanted='venturi-shared-input-fixtures-v1/build/77fedac214853798/libventuri_blocks.so' if e['shared_inputs'] else 'venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'
  assert wanted in (raw/'campaign-v3'/e['label']/'loaded-maps.txt').read_text()
 assert all(w['emitted_tokens']==32*w['request_count'] and all(q['response']['tokens']==r['case']['expected_generated_ids'] for q in w['requests']) for w in r['waves'])
proof=read(raw/'final-v4/executed-source.json')
for rel,h in proof.items():assert sha(root/rel)==h,rel
assert campaign['source_sha256']==read(out/'remote-terminal-audit.json')['executed_snapshots']['shared-input-src-v2']
terminal=read(out/'remote-terminal-audit.json');assert not terminal['owned_inference_processes'] and not terminal['listening_port_18948'];assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy'];assert '3651 MiB' in terminal['gpu_state'] and '3189 MiB' in terminal['gpu_state'];assert all(w['exit_code']==0 and w['remaining_process_state'] is None for w in terminal['workers'].values())
snapshot_matches={}
for snapshot,files in terminal['executed_snapshots'].items():
 delta=[]
 for rel,h in files.items():
  current=root/rel
  if current.is_file() and sha(current)==h:continue
  preserved=out/'executed-source-deltas'/snapshot/rel;assert preserved.is_file() and sha(preserved)==h,(snapshot,rel);delta.append(rel)
 snapshot_matches[snapshot]=dict(total=len(files),preserved_prior_python_files=delta)
build=read(next((raw/'fixtures-v1/build').rglob('build.json')));assert build['architecture']=='sm70';assert len(build['source_sha256'])==6
for rel,h in build['source_sha256'].items():assert sha(root/'src/venturi/ops/csrc'/rel)==h
for record in [read(raw/'fixtures-v1/completed.json'),read(raw/'qualification-v2/completed.json'),read(raw/'final-v4/completed.json')]:assert record['completed'] and record['sha256']==build['library_sha256']
assert build['library_sha256']==campaign['candidate_library_sha256']==qualified['candidate_library_sha256']==mtp['library_sha256']
assert campaign['baseline_library_sha256']==qualified['baseline_library_sha256']
assert sha(raw/'qualification-v2/qualification/report.json')==campaign['qualification_report_sha256']
resources=read(raw/'final-v4/resources.json');assert resources['library_sha256']==build['library_sha256'] and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
for name in ['fp16-c1','fp16-c8','q8-residual-c1','q8-residual-c8']:
 m=read(raw/'final-v4'/(name+'-metadata.json'));assert m['profiled'] and m['logical_requests']==(8 if name.endswith('c8') else 1)
 d=read(raw/'final-v4'/(name+'-attribution.json'));assert len(d['cohorts'])==1 and d['cohorts'][0]['requests']==1
prior={}
for cap in sorted((root/'benchmarks').iterdir()):
 if cap==out or not cap.is_dir():continue
 record={}
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  path=cap/name
  if not path.exists():continue
  m=read(path);m=m.get('files',m);n=0;missing=[]
  for rel,h in m.items():
   p=base/rel
   if not p.exists() and cap.name=='decoder-layer':p=base/pathlib.Path(rel).name
   if p.is_file():assert sha(p)==h,str(p);n+=1
   else:missing.append(rel)
  if missing:assert cap.name=='decoder-layer' and n==4
  record[name]=dict(verified=n,legacy_unavailable_files=missing)
 if record:prior[cap.name]=record
assert len(prior)==20,len(prior)
shutil.copy2('/tmp/venturi-shared-input-freeze.py',out/'launchers/freeze.py')
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools','benchmarks/v100-shared-inputs/summarize.py'],cwd=root,stdout=(out/'ruff.log').open('w'),check=True);subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
metrics=read(out/'metrics.json');assert not metrics['external_performance_goal_complete']
assert all(v<1 for v in metrics['shared_residual_vs_fresh_llama_non_speculative_ratio'].values())
for folder in ['src/venturi/models','src/venturi/ops']:
 for old in (root/'benchmarks/v100-native-residual/source'/folder).rglob('*.py'):
  assert sha(old)==sha(root/old.relative_to(root/'benchmarks/v100-native-residual/source'))
assert all(v>1 for name,rows in metrics['same_profile_sharing_speedup'].items() if name.startswith('repeat') for v in rows.values())
for profile in ['fp16','q8-residual']:
 for count in [1,8]:
  old=metrics['profiles'][f'unshared-{profile}-c{count}']['attribution']['cohorts'][0];new=metrics['profiles'][f'shared-{profile}-c{count}']['attribution']['cohorts'][0]
  for direction in ['host_to_device','device_to_host']:assert old['transfers'][direction]['bytes']-new['transfers'][direction]['bytes']==44236800*count
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 p=root/name
 if p.exists():files.append(p)
files += [out/'summary.md',out/'summarize.py'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_current_source_files_verified=len(proof),executed_snapshot_matches=snapshot_matches,native_library_sha256=build['library_sha256'],native_six_build_sources_verified=True,default_fp16_arithmetic_and_fixed_byte_gates_preserved=True,arithmetic_profiles_unchanged_by_sharing=True,trained_fixed_cpu_and_original_trajectory_gates_passed=True,broad_original_bf16_corpus_quality_qualified=False,same_profile_scalar_batch_mtp_state_exact=True,prior_reference_pins_verified=True,three_repeat_promoted_matrix_complete=True,all_measured_external_cells_still_lose=True,all_owned_workers_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,hardware_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,local_rust_compilation=False,prior_capsules_unchanged=prior)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n');evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),goal_complete=False)))
