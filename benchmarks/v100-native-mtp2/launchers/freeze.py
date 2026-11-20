from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
root=Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-native-mtp2';raw=out/'v100'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
t=read(out/'terminal-audit.json');assert not t['owned_inference_processes'] and not t['listening_ports'];assert 'RmProfilingAdminOnly: 1' in t['counter_policy'];assert len(t['workers'])==14
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
prior_caps=read(out/'prior-capsule-audit.json');assert len(prior_caps)==28
for cap_name in prior_caps:
 cap=root/'benchmarks'/cap_name
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  p=cap/name
  if not p.exists():continue
  manifest=read(p);manifest=manifest.get('files',manifest);n=0;missing=[]
  for rel,h in manifest.items():
   p=base/rel
   if not p.exists() and cap_name=='decoder-layer':p=base/Path(rel).name
   if p.is_file():assert sha(p)==h,str(p);n+=1
   else:missing.append(rel)
  if missing:assert cap_name=='decoder-layer' and n==4
assert '776 passed, 1444 skipped' in (out/'local-final-tests.log').read_text()
assert '1768 passed, 448 skipped' in (raw/'prefix-v5/full-tests.log').read_text()
assert '1772 passed, 448 skipped' in (raw/'prefix-tests-v13/full-tests.log').read_text()
assert '113 passed, 27 skipped' in (raw/'prefix-tests-v13/mtp-tests.log').read_text()
for stage,recovery,libhash in [('model-v2','replay','17e64d758d64cf682b784a14bf80b4844398877628e9239c3b38420bb13afab1'),('prefix-model-v6','prefix','9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894')]:
 q=read(raw/stage/'qualification/report.json');assert q['completed'] and q['draft_tokens']==2 and q.get('target_recovery','replay')==recovery and q['library_sha256']==libhash and len(q['cases'])==4 and len(q['recurrence_reference'])==3 and all(x['passed'] for x in q['recurrence_reference'])
 for c in q['cases']:
  assert c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['committed_draft_state_equal_mtp1'] and c['pinned_target_trajectory_state_reused'] and c['mtp_stats']['forced_proposals']==0
  if recovery=='prefix':assert c['mtp_stats']['replayed_input_tokens']==0 and c['mtp_stats']['prefix_trace_extra_primary_bytes']==153944064
assert "KeyError: 'recovery'" in (raw/'prefix-model-v6/failure.txt').read_text()
r=read(raw/'prefix-model-recovery-v9/recovery.json');assert r['prior_report_sha256']==sha(raw/'prefix-model-v6/qualification/report.json')==sha(raw/'prefix-model-recovery-v9/qualification/report.json')
for stage,experiments in [('campaign-v3',3),('prefix-campaign-v10',6)]:
 c=read(raw/stage/'campaign-report.json');assert c['completed'] and len(c['experiments'])==experiments
 for e in c['experiments']:
  assert e['completed'] and e['stable_outputs'];r=read(raw/stage/e['label']/'timed/report.json');assert r['completed']
  counts=[1,8] if e['label'].startswith('screen') else [1,2,4,8];repeats=1 if e['label'].startswith('screen') else 3
  assert [s['request_count'] for s in e['summary']]==counts and all(s['trials']==repeats for s in e['summary'])
  assert len(r['waves'])==len(counts)*repeats
  for w in r['waves']:
   assert w['emitted_tokens']==32*w['request_count']
   assert all(x['response']['tokens']==r['case']['expected_generated_ids'] for x in w['requests'])
   if e.get('mtp_recovery')=='prefix':assert all(x['response']['mtp_stats']['replayed_input_tokens']==0 for x in w['requests'])
 assert c['baseline_library_sha256']=='e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a'
assert not read(raw/'campaign-v3/campaign-report.json')['promoted']
c=read(raw/'prefix-campaign-v10/campaign-report.json');assert c['promoted'] and c['no_default_promotion']
assert c['qualification_report_sha256']==sha(raw/'prefix-model-recovery-v9/qualification/report.json')
assert t['executed_snapshots']==read(out/'executed-snapshot-hashes.json')
resolved={}
for snap,files in t['executed_snapshots'].items():
 delta=[]
 for rel,h in files.items():
  if (root/rel).is_file() and sha(root/rel)==h:continue
  p=out/'executed-source-deltas'/snap/rel;assert p.is_file() and sha(p)==h,(snap,rel);delta.append(rel)
 resolved[snap]=dict(total=len(files),preserved_executed_delta=delta)
assert read(raw/'prefix-tests-v13/executed-source.json')==t['executed_snapshots']['native-mtp2-src-v6']
assert c['source_sha256']==t['executed_snapshots']['native-mtp2-src-v4']
b=read(next((raw/'prefix-v5/build').glob('*/build.json')));assert b['library_sha256']==c['candidate_library_sha256'] and len(b['source_sha256'])==9
for rel,h in b['source_sha256'].items():assert sha(root/'src/venturi/ops/csrc'/rel)==h
for stage,names,hash_ in [('final-v4',['mtp1-c1','mtp1-c8','mtp2-c1','mtp2-c8'],'17e64d758d64cf682b784a14bf80b4844398877628e9239c3b38420bb13afab1'),('prefix-final-v11',['mtp2-prefix-c1','mtp2-prefix-c8'],b['library_sha256']),('full-profile-v14',['mtp2-replay-full-c1','mtp2-replay-full-c8','mtp2-prefix-full-c1','mtp2-prefix-full-c8'],b['library_sha256'])]:
 resources=read(raw/stage/'resources.json');assert resources['library_sha256']==hash_ and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
 for name in names:
  p=read(raw/stage/(name+'-metadata.json'));assert p['profiled'] and p['projection_rows']==32 and p['library_sha256']==hash_
  a=read(raw/stage/(name+'-attribution.json'))['cohorts'][0];assert a['requests']==p['logical_requests']
  if 'prefix' in name:assert any('gdn_prefix' in k['name'] for k in a['kernel_by_name'])
 assert (raw/stage/'exit-code.txt').read_text().strip()=='0'
 if stage=='full-profile-v14':
  for name in names:
   p=read(raw/stage/(name+'-metadata.json'));assert all(s['rollback_calls']>0 and s['replayed_input_tokens']==(0 if 'prefix' in name else 12) for s in p['stats']);assert all(len(q['ids'])==32 for q in p['instrumented_wave']['requests'])
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools',str(out/'summarize.py')],cwd=root,stdout=(out/'ruff.log').open('w'),check=True)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
m=read(out/'metrics.json');assert m['full_matrix_executed'] and not m['external_speed_goal_complete']
shutil.copy2(__file__,out/'launchers/freeze.py');files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 if (root/name).exists():files.append(root/name)
files += [out/'summary.md',out/'summarize.py',out/'design.md',out/'source-notes.md',out/'next-bottlenecks.md'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_snapshots_verified=resolved,library_sha256=b['library_sha256'],negative_replay_experiment_preserved=True,prefix_replay_inputs_zero=True,extra_recurrent_snapshot_bytes_per_primary_slot=153944064,projection_budget=32,slot_context_capacity=256,default_mtp_depth_still1=True,recovery_default_still_replay=True,target_and_draft_state_gates_passed=True,independent_reference_thresholds_unchanged=True,old_98_vector_target_reference_explicitly_reused=True,broad_bf16_corpus_quality_qualified=False,three_repeat_promoted_matrix_complete=True,external_controls_explicitly_reused=True,external_five_percent_acceptance_incomplete=True,reporting_only_failure_and_recovery_preserved=True,all_owned_processes_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,driver_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,no_local_rust_or_cuda_compilation=True,prior_capsules_unchanged=prior_caps)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=28,goal_complete=False)))
