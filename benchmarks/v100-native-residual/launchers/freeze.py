import pathlib,json,hashlib,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-native-residual';raw=out/'v100'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'evidence-sha256.json').exists(),'Already frozen'
for name in ['fixtures-v1','qualification-v2','campaign-v3','final-v4']:assert int((raw/name/'exit-code.txt').read_text())==0,name
assert '1020 passed, 440 skipped' in (raw/'final-v4/tests.log').read_text()
assert '760 passed, 700 skipped' in (out/'local-tests.log').read_text()
assert '24 passed' in (raw/'qualification-v2/profile-tests.log').read_text()
assert '130 passed, 125 skipped' in (raw/'fixtures-v1/default-regression-tests.log').read_text()
qualified=read(raw/'qualification-v2/qualification/report.json');assert qualified['completed'] and not qualified['full_original_bf16_corpus_quality_qualified']
assert qualified['fixed_logit_thresholds']==dict(relative_l2=.02,centered_cosine=.999,reference_kl_to_native=.002,top1_equal=True)
for name,p in qualified['profiles'].items():
 assert p['passed'] and len(p['fixed_cpu_steps'])==2 and len(p['trained_cases'])==4
 for step in p['fixed_cpu_steps']:
  assert step['logits_vs_pinned_cpu']['passed']
  if name=='fp16':assert step['original_native_logit_bytes_equal'] and all(s['bytes_equal'] for s in step['original_hidden_state_comparisons'])
 for case in p['trained_cases']:
  assert case['original_ids']==case['generated_ids'] and case['native_generation_without_torch_math'] and case['same_profile_diagnostic_generate_state_bytes_exact']
  assert all(c['passed'] for c in case['teacher_forced_logit_checks'])
 if name=='q8-residual':assert p['same_profile_batch']['original_ids_equal'] and p['same_profile_batch']['all_eight_private_cache_bytes_match_scalar']
 else:assert p['same_profile_batch'] is None
assert sum(len(c['teacher_forced_logit_checks']) for c in qualified['profiles']['q8-residual']['trained_cases'])==98
reuse=read(out/'qualification-reuse.json');ref=root/reuse['prior_report']['path'];assert sha(ref)==reuse['prior_report']['sha256'];draft=root/reuse['draft_cuda_source']['current'];assert sha(draft)==reuse['draft_cuda_source']['sha256']==sha(root/reuse['draft_cuda_source']['prior'])
mtp=read(raw/'campaign-v3/mtp-qualification/report.json');assert mtp['completed'] and mtp['projection_profile']=='q8-residual' and mtp['mtp_draft_projection_profile']=='fp16'
assert len(mtp['cases'])==4 and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] for c in mtp['cases'])
campaign=read(raw/'campaign-v3/campaign-report.json');assert campaign['completed'] and len(campaign['experiments'])==10
for e in campaign['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v3'/e['label']/'timed/report.json');assert r['completed']
 if e['label'].startswith('repeat'):
  counts=[1,2,4,8] if e['label'] in ['repeat-original','repeat-residual','repeat-llama-tensor-off'] else [1]
  assert [s['request_count'] for s in e['summary']]==counts and all(s['trials']==3 for s in e['summary'])
  assert len(r['waves'])==len(counts)*3
 if e['engine']=='venturi':assert 'venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so' in (raw/'campaign-v3'/e['label']/'loaded-maps.txt').read_text()
 assert all(w['emitted_tokens']==32*w['request_count'] and all(q['response']['tokens']==r['case']['expected_generated_ids'] for q in w['requests']) for w in r['waves'])
proof=read(raw/'final-v4/executed-source.json')
for rel,h in proof.items():assert sha(root/rel)==h,rel
assert campaign['source_sha256']==read(out/'remote-terminal-audit.json')['executed_snapshots']['model-profile-src-v4']
terminal=read(out/'remote-terminal-audit.json');assert not terminal['owned_inference_processes'] and not terminal['listening_port_18947'];assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy'];assert '3651 MiB' in terminal['gpu_state'] and '3189 MiB' in terminal['gpu_state'];assert all(w['exit_code']==0 and w['remaining_process_state'] is None for w in terminal['workers'].values())
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
assert build['library_sha256']==campaign['library_sha256']==qualified['library_sha256']==mtp['library_sha256']
assert sha(raw/'qualification-v2/qualification/report.json')==campaign['qualification_report_sha256']
resources=read(raw/'final-v4/resources.json');assert resources['library_sha256']==build['library_sha256'] and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
for name in ['fp16-c1','fp16-c8','q8-residual-c1','q8-residual-c8']:
 m=read(raw/'final-v4'/(name+'-metadata.json'));assert m['profiled'] and m['logical_requests']==(8 if name.endswith('c8') else 1)
 d=read(raw/'final-v4'/(name+'-attribution.json'));assert len(d['cohorts'])==1 and d['cohorts'][0]['requests']==1
smoke=(raw/'final-v4/cli-smoke.json').read_text();smoke=json.loads(smoke[smoke.index('{'):]);assert smoke['projection_profile']=='q8-residual' and smoke['native_memory']['activation_quantization_passes']==2 and len(smoke['generated_ids'])==8
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
assert len(prior)==19,len(prior)
shutil.copy2('/tmp/venturi-residual-freeze.py',out/'launchers/freeze.py')
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools','benchmarks/v100-native-residual/summarize.py'],cwd=root,stdout=(out/'ruff.log').open('w'),check=True);subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
metrics=read(out/'metrics.json');assert not metrics['external_performance_goal_complete'];assert all(v>1 for v in metrics['residual_vs_original_speedup'].values()) and all(v<1 for v in metrics['residual_vs_llama_non_speculative_ratio'].values())
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
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_current_source_files_verified=len(proof),executed_snapshot_matches=snapshot_matches,native_library_sha256=build['library_sha256'],native_six_build_sources_verified=True,default_fp16_arithmetic_and_fixed_byte_gates_preserved=True,new_profile_explicit_changed_arithmetic=True,trained_fixed_cpu_and_original_trajectory_gates_passed=True,broad_original_bf16_corpus_quality_qualified=False,same_profile_scalar_batch_mtp_state_exact=True,unchanged_draft_reference_hash_verified=True,three_repeat_promoted_matrix_complete=True,all_measured_external_cells_still_lose=True,all_owned_workers_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,hardware_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,local_rust_compilation=False,prior_capsules_unchanged=prior)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n');evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),goal_complete=False)))
