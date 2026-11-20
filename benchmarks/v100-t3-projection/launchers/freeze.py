from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
root=Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-t3-projection';raw=out/'v100'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
t=read(out/'terminal-audit.json');assert not t['owned_inference_processes'] and not t['listening_ports'] and not t['remaining_periodic_nvml_samplers'];assert 'RmProfilingAdminOnly: 1' in t['counter_policy'] and len(t['workers'])==4
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
prior_caps=read(out/'prior-capsule-audit.json');assert len(prior_caps)==29
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
for stage in ['study-v1','model-v2']:
 assert '1906 passed, 448 skipped' in (raw/stage/'full-tests.log').read_text()
 assert '445 passed' in (raw/stage/'mtp-tests.log').read_text()
 assert (raw/stage/'exit-code.txt').read_text().strip()=='0'
assert '776 passed, 1578 skipped' in (out/'local-v2-tests.log').read_text()
op=read(raw/'study-v1/operators/report.json');assert op['completed'] and len(op['cases'])==11 and not op['profiled']
for c in op['cases']:
 assert c['all_output_bytes_equal'] and len(c['waves']) in [15,24] and all(v['relative_l2']<.001 for v in c['sampled_fp64'])
q=read(raw/'model-v2/qualification/report.json');assert q['completed'] and q['profiles']['q8-residual']['same_profile_bytes_equal'] and q['fixed_work']['max_tokens']==256
p=q['profiles']['q8-residual']['candidate'];assert len(p['cases'])==4 and sum(len(x['all_logit_bytes']) for x in p['cases'])==98 and len(p['batch']['private_cache'])==len(p['reset_continuation']['private_cache'])==8
m=read(raw/'model-v2/mtp-qualification/report.json');assert m['completed'] and m['target_recovery']=='prefix' and len(m['cases'])==4 and len(m['recurrence_reference'])==3 and all(v['passed'] for v in m['recurrence_reference'])
assert all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['committed_draft_state_equal_mtp1'] and c['pinned_target_trajectory_state_reused'] and c['mtp_stats']['forced_proposals']==c['mtp_stats']['replayed_input_tokens']==0 for c in m['cases'])
c=read(raw/'campaign-v3/campaign-report.json');assert c['completed'] and not c['promoted'] and not c['batch_promoted'];assert c['baseline_library_sha256']=='9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894'
for e in c['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v3'/e['label']/'timed/report.json');assert r['completed']
 counts=[1,8] if e['label'].startswith('screen') else ([1] if e['engine']=='llama' else [1,2,4,8]);repeats=1 if e['label'].startswith('screen') else 3
 assert [v['request_count'] for v in e['summary']]==counts and all(v['trials']==repeats for v in e['summary'])
 for w in r['waves']:
  assert w['emitted_tokens']==32*w['request_count'] and all(x['response']['tokens']==r['case']['expected_generated_ids'] for x in w['requests'])
  if e['mode']=='batch':assert {x['response']['batch_request_count'] for x in w['requests']}=={w['request_count']} and len({x['response']['batch_cohort_id'] for x in w['requests']})==1
  if e['mtp_depth'] and e['engine']=='venturi':assert all(x['response']['mtp_stats']['replayed_input_tokens']==0 for x in w['requests'])
resolved={}
for snap,files in t['executed_snapshots'].items():
 delta=[]
 for rel,h in files.items():
  if (root/rel).is_file() and sha(root/rel)==h:continue
  p=out/'executed-source-deltas'/snap/rel;assert p.is_file() and sha(p)==h,(snap,rel);delta.append(rel)
 resolved[snap]=dict(total=len(files),preserved_executed_delta=delta)
assert c['source_sha256']==read(raw/'model-v2/executed-source.json')==t['executed_snapshots']['t3-projection-src-v2']
study_build=read(next((raw/'study-v1/build').glob('*/build.json')))
for rel,h in study_build['source_sha256'].items():assert sha(root/'src/venturi/ops/csrc'/rel)==h
assert not read(out/'decision.json')['promote_to_model_default']
assert read(out/'decision.json')['screen_complete_request_ratio']==c['screen_gains']
assert read(out/'restored-dispatch-proof.json')['model_dispatch_restored']
b=read(next((raw/'model-v2/build').glob('*/build.json')));assert b['library_sha256']==q['candidate_library_sha256']==m['library_sha256']==c['candidate_library_sha256'] and len(b['source_sha256'])==9
resources=read(raw/'final-v4/resources.json');assert resources['library_sha256']==c['candidate_library_sha256'] and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
for count in [1,8]:
 p=read(raw/'final-v4'/f'mtp2-prefix-c{count}-metadata.json');assert p['profiled'] and p['projection_rows']==32 and p['logical_requests']==count
 assert all(s['replayed_input_tokens']==0 and s['rollback_calls']>0 for s in p['stats'])
 a=read(raw/'final-v4'/f'mtp2-prefix-c{count}-attribution.json')['cohorts'][0];assert any('tiled_dot<(int)3' in k['name'] for k in a['kernel_by_name'])
for name,rec in read(raw/'final-v4/reused-control-profiles.json').items():assert rec['sha256']==sha(root/'benchmarks/v100-native-mtp2/v100/full-profile-v14'/name)
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools',str(out/'summarize.py')],cwd=root,stdout=(out/'ruff.log').open('w'),check=True);subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True);metrics=read(out/'metrics.json');assert not metrics['external_speed_goal_complete']
shutil.copy2(__file__,out/'launchers/freeze.py');files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 if (root/name).exists():files.append(root/name)
files += [out/'summary.md',out/'summarize.py',out/'design.md',out/'next-bottlenecks.md'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_snapshots_verified=resolved,library_sha256=c['candidate_library_sha256'],full_sharing_regressions_preserved=True,projection_arithmetic_and_allocation_unchanged=True,new_token3_row8_research_kernel=True,model_dispatch_not_promoted=True,model_dispatch_restored=True,fresh_trained_logits_and_private_state_exact=True,independent_reference_thresholds_unchanged=True,old_98_vector_cpu_quality_reference_explicitly_reused=True,broad_bf16_corpus_quality_qualified=False,three_repeat_mtp_matrix_executed=c['promoted'],three_repeat_batch_matrix_executed=c['batch_promoted'],stronger_controls_explicitly_reused=True,external_speed_goal_complete=False,all_owned_processes_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,driver_counter_policy_unchanged=True,no_commit_or_push=True,no_local_rust_or_cuda_compilation=True,prior_capsules_unchanged=prior_caps)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=29,goal_complete=False)))
