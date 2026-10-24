from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
root=Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-prefill-budget';raw=out/'v100'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
t=read(out/'terminal-audit.json');assert not t['owned_inference_processes'] and not t['listening_ports'];assert 'RmProfilingAdminOnly: 1' in t['counter_policy']
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
prior_caps=read(out/'prior-capsule-audit.json');assert len(prior_caps)==27
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
assert '769 passed, 1350 skipped' in (out/'local-tests.log').read_text()
assert '35 passed' in (raw/'model-v1/budget-tests.log').read_text()
assert '1679 passed, 440 skipped' in (raw/'model-v1/full-tests.log').read_text()
prior_path=root/'benchmarks/v100-gdn-compact/v100/model-v2/qualification/report.json';prior=read(prior_path)
for rows in [128,256]:
 q=read(raw/f'model-v1/qualification-{rows}/report.json');assert q['completed'] and q['executed_candidate_profiles']==['q8-residual'] and not q['broad_bf16_corpus_quality_qualified']
 assert q['prior_report_sha256']==sha(prior_path) and q['fixed_work']['max_tokens']==rows and q['fixed_work']['control_max_tokens']==32
 p=q['profiles']['q8-residual'];assert p['same_profile_bytes_equal'] and p['candidate']==p['control']==prior['profiles']['q8-residual']['candidate']
 assert sum(len(x['all_logit_bytes']) for x in p['candidate']['cases'])==98
 assert len(p['candidate']['batch']['private_cache'])==len(p['candidate']['reset_continuation']['private_cache'])==8
 assert list(q['unexecuted_profile_reference'])==['fp16']
mtp=read(raw/'model-v1/unchanged-mtp-reference.json');assert mtp['sha256']==sha(root/'benchmarks/v100-gdn-compact/v100/model-v2/mtp-qualification/report.json')
assert sha(root/'tools/qualify_qwen_mtp.py')==sha(root/'benchmarks/v100-gdn-compact/source/tools/qualify_qwen_mtp.py')
c=read(raw/'campaign-v2/campaign-report.json');assert c['completed'] and c['promoted'] and c['promoted_projection_rows'] in [128,256] and len(c['experiments'])==5
for e in c['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v2'/e['label']/'timed/report.json');assert r['completed']
 counts=[1,8] if e['label'].startswith('screen') else [1,2,4,8];repeats=1 if e['label'].startswith('screen') else 3
 assert [s['request_count'] for s in e['summary']]==counts and all(s['trials']==repeats for s in e['summary'])
 assert len(r['waves'])==len(counts)*repeats
 for w in r['waves']:
  assert w['emitted_tokens']==32*w['request_count']
  assert all(x['response']['tokens']==r['case']['expected_generated_ids'] for x in w['requests'])
  assert len({x['response']['batch_cohort_id'] for x in w['requests']})==1
 assert 'venturi-gdn-compact-model-v2' in (raw/'campaign-v2'/e['label']/'loaded-maps.txt').read_text()
 assert e['health']['projection_rows']==e['projection_rows']
for q in c['qualification_gates']:assert q['sha256']==sha(raw/f'model-v1/qualification-{q["projection_rows"]}/report.json')
assert c['baseline_library_sha256']==c['candidate_library_sha256']=='e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a'
proof=read(raw/'final-v3/executed-source.json');assert proof==c['source_sha256']==t['executed_snapshots']['prefill-budget-src-v1']
resolved={}
for snap,files in t['executed_snapshots'].items():
 delta=[]
 for rel,h in files.items():
  if (root/rel).is_file() and sha(root/rel)==h:continue
  p=out/'executed-source-deltas'/snap/rel;assert p.is_file() and sha(p)==h,(snap,rel);delta.append(rel)
 resolved[snap]=dict(total=len(files),preserved_executed_delta=delta)
b=read(raw/'model-v1/reused-build.json');assert b['sha256']==c['candidate_library_sha256'] and len(b['build']['source_sha256'])==9
for rel,h in b['build']['source_sha256'].items():assert sha(root/'src/venturi/ops/csrc'/rel)==h==sha(root/'benchmarks/v100-gdn-compact/source/src/venturi/ops/csrc'/rel)
for name,record in read(raw/'final-v3/reused-control-profiles.json').items():assert record['sha256']==sha(root/'benchmarks/v100-gdn-compact/v100/final-v5'/name)
resources=read(raw/'final-v3/resources.json');assert resources['library_sha256']==c['candidate_library_sha256'] and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
for count in [1,8]:
 p=read(raw/'final-v3'/f'candidate-c{count}-metadata.json');assert p['profiled'] and p['projection_rows']==c['promoted_projection_rows']
 a=read(raw/'final-v3'/f'candidate-c{count}-attribution.json')['cohorts'][0];assert a['requests']==1
 assert any('gdn_compact' in k['name'] for k in a['kernel_by_name'])
assert (raw/'final-v3/exit-code.txt').read_text().strip()=='0'
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools',str(out/'summarize.py')],cwd=root,stdout=(out/'ruff.log').open('w'),check=True)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
m=read(out/'metrics.json');assert m['candidate_promoted'] and not m['external_performance_goal_complete'] and not all(x>=1.05 for x in m['candidate_vs_reused_external_non_spec_ratio'].values())
assert m['same_profile_speedup']['8']>1.025
shutil.copy2(__file__,out/'launchers/freeze.py');files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 if (root/name).exists():files.append(root/name)
files += [out/'summary.md',out/'summarize.py',out/'design.md',out/'next-bottlenecks.md'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_snapshots_verified=resolved,library_sha256=c['candidate_library_sha256'],native_binary_unchanged=True,projection_budget=c['promoted_projection_rows'],slot_context_capacity=256,default_adapter_budget_still32=True,no_new_kernel_or_architecture_claim=True,trained_logit_cache_bytes_exact_at128_and256=True,independent_reference_thresholds_unchanged=True,mtp_serving_and_measurements_unchanged_explicitly_reused=True,broad_bf16_corpus_quality_qualified=False,three_repeat_promoted_matrix_complete=True,external_controls_explicitly_reused=True,external_five_percent_acceptance_incomplete=True,all_owned_processes_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,driver_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,no_local_rust_or_cuda_compilation=True,prior_capsules_unchanged=prior_caps)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=27,goal_complete=False)))
