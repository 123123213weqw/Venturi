import pathlib,json,hashlib,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-strided-output';raw=out/'v100';proto=out/'experiments/direct-primary-output'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def executed(rel,h,snap):
 for p in [root/rel,out/'executed-source-deltas'/snap/rel,proto/rel]:
  if p.is_file() and sha(p)==h:return str(p.relative_to(root))
 raise AssertionError((snap,rel,h))
assert not (out/'evidence-sha256.json').exists(),'Already frozen'
terminal=read(out/'remote-terminal-audit.json');assert not terminal['owned_inference_processes'] and not terminal['listening_ports'];assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy'];assert '3651 MiB' in terminal['gpu_state'] and '3189 MiB' in terminal['gpu_state']
for n in ['fixtures-v3','campaign-v5','final-v8','external-depth-v9']:assert int((raw/n/'exit-code.txt').read_text())==0 and terminal['workers'][n]['exit_code']==0
assert int((raw/'fixtures-v1/exit-code.txt').read_text())==1
assert '1280 passed, 440 skipped' in (raw/'final-v8/tests.log').read_text()
assert '761 passed, 885 skipped' in (out/'local-tests.log').read_text()
assert '761 passed, 959 skipped' in (out/'local-prototype-tests.log').read_text()
assert '166 passed' in (raw/'fixtures-v3/profile-tests.log').read_text()
assert '130 passed, 125 skipped' in (raw/'fixtures-v3/default-regression-tests.log').read_text()
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
q=read(raw/'fixtures-v3/qualification/report.json');assert q['completed'] and not q['broad_bf16_corpus_quality_qualified']
for controls in q['profiles'].values():
 assert controls['same_profile_bytes_equal'] and controls['control']==controls['candidate']
 for label in ['control','candidate']:
  p=controls[label];assert p['passed'] and len(p['cases'])==4
  assert sum(len(c['all_logit_bytes']) for c in p['cases'])==98
  assert all(c['native_generate_without_torch_math'] and c['diagnostic_generate_state_bytes_equal'] for c in p['cases'])
  assert len(p['batch']['private_cache'])==8 and len(p['reset_continuation']['private_cache'])==8
assert q['prior_report_sha256']==sha(root/'benchmarks/v100-q8-wmma/v100/guarded-v4/qualification/report.json')
mtp=read(raw/'fixtures-v3/mtp-qualification/report.json');assert mtp['completed'] and len(mtp['cases'])==4 and mtp['projection_profile']=='q8-residual' and mtp['mtp_draft_projection_profile']=='fp16';assert not mtp['block_reference']
assert all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['native_without_python_math'] and c['mtp_stats']['forced_proposals']==0 for c in mtp['cases'])
ref=read(raw/'fixtures-v3/reused-draft-reference.json');assert ref['sha256']==sha(root/'benchmarks/v100-shared-inputs/v100/campaign-v3/mtp-qualification/report.json')
campaign=read(raw/'campaign-v5/campaign-report.json');assert campaign['completed'] and not campaign['promoted'] and len(campaign['experiments'])==2
assert .98<campaign['screen_gains']['1']<1 and .99<campaign['screen_gains']['8']<1.01
for e in campaign['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v5'/e['label']/'timed/report.json');assert r['completed'] and len(r['waves'])==2
 assert [s['request_count'] for s in e['summary']]==[1,8] and all(s['trials']==1 for s in e['summary'])
 assert all(w['emitted_tokens']==32*w['request_count'] and all(x['response']['tokens']==r['case']['expected_generated_ids'] for x in w['requests']) for w in r['waves'])
 assert ('venturi-wmma-guarded-v4' if e['label']=='screen-control' else 'venturi-strided-fixtures-v1') in (raw/'campaign-v5'/e['label']/'loaded-maps.txt').read_text()
assert campaign['candidate_library_sha256']==q['candidate_library_sha256']==mtp['library_sha256']
assert campaign['baseline_library_sha256']==q['baseline_library_sha256']
assert campaign['qualification_report_sha256']==sha(raw/'fixtures-v3/qualification/report.json')
assert campaign['fresh_trained_mtp_state_qualification']['sha256']==sha(raw/'fixtures-v3/mtp-qualification/report.json')
proof=read(raw/'final-v8/executed-source.json');assert proof==campaign['source_sha256']==terminal['executed_snapshots']['strided-src-v2']
resolved={snap:{rel:executed(rel,h,snap) for rel,h in files.items()} for snap,files in terminal['executed_snapshots'].items()}
b=read(next((raw/'fixtures-v1/build').rglob('build.json')));assert b['architecture']=='sm70' and len(b['source_sha256'])==7
for rel,h in b['source_sha256'].items():executed('src/venturi/ops/csrc/'+rel,h,'strided-src-v1')
assert b['library_sha256']==campaign['candidate_library_sha256']==read(raw/'fixtures-v3/reused-build.json')['library_sha256']
resources=read(raw/'final-v8/resources.json');assert resources['library_sha256']==b['library_sha256'] and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
for c in [1,8]:
 m=read(raw/'final-v8'/f'q8-residual-c{c}-metadata.json');assert m['profiled'] and m['logical_requests']==c
 a=read(raw/'final-v8'/f'q8-residual-c{c}-attribution.json');assert len(a['cohorts'])==1 and a['cohorts'][0]['requests']==1
 assert not any('vwt::' in k['name'] for k in a['cohorts'][0]['kernel_by_name'])
copy=read(raw/'final-v8/copy-api-partition.json')
for c in [1,8]:
 old=copy[f'control-c{c}']['groups'];new=copy[f'candidate-c{c}']['groups']
 assert old['cudaMemcpy2DAsync_v3020/kind=8']['count']>0
 assert not any('Memcpy2DAsync' in key and key.endswith('kind=8') for key in new)
external=read(raw/'external-depth-v9/campaign-report.json');assert external['completed'] and len(external['experiments'])==3
assert external['reused_control']['sha256']==sha(root/'benchmarks/v100-q8-wmma/v100/campaign-v5/campaign-report.json')
assert external['source_sha256']==proof
for e in external['experiments']:
 assert e['mtp_depth'] in [2,3,4]
 if not e['completed']:assert e.get('failure');continue
 assert e['stable_outputs'];r=read(raw/'external-depth-v9'/e['label']/'timed/report.json');assert r['completed'] and len(r['waves'])==3
 assert all(w['request_count']==1 and w['emitted_tokens']==32 and w['requests'][0]['response']['tokens']==r['case']['expected_generated_ids'] for w in r['waves'])
restore=read(out/'production-restore.json');assert restore['restored'] and not restore['prototype_enabled_in_model']
assert restore['production_library_sha256']==campaign['baseline_library_sha256']
for rel,h in restore['native_source_sha256'].items():assert sha(root/'src/venturi/ops/csrc'/rel)==h
assert not (root/'tests/test_q8_dp4a_strided.py').exists()
# Compare the arithmetic body, excluding the explicitly changed address/signature.
old=(root/'src/venturi/ops/csrc/q8_dp4a.cuh').read_text();new=(proto/'src/venturi/ops/csrc/q8_dp4a.cuh').read_text();old=old[old.index('template<int Tokens'):old.index('template<int Passes>void launch')];new=new[new.index('template<int Tokens'):new.index('template<int Passes,bool Strided=false>void launch')];new=new.replace(',bool Strided=false','').replace('int groups,int output_stride','int groups').replace('(Strided?output_stride:n)','n');assert old==new
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
assert len(prior)==22,len(prior)
shutil.copy2(__file__,out/'launchers/freeze.py')
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools',str(out/'summarize.py')],cwd=root,stdout=(out/'ruff.log').open('w'),check=True);subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
metrics=read(out/'metrics.json');assert not metrics['external_performance_goal_complete'] and not metrics['candidate_promoted']
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 if (root/name).exists():files.append(root/name)
files += [out/'summary.md',out/'summarize.py'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_prototype_source_resolution=resolved,production_native_restored_exact=True,production_library_sha256=restore['production_library_sha256'],rejected_prototype_library_sha256=b['library_sha256'],arithmetic_body_preserved=True,trained_scalar_batch_and_mtp_state_gates_passed=True,broad_original_bf16_corpus_quality_qualified=False,independent_draft_cpu_reference_explicitly_reused=True,copy_removal_verified_without_speedup=True,candidate_not_promoted=True,no_repeat_promotion_campaign_for_failed_screen=True,external_mtp_depths_tested=[2,3,4],best_tested_configuration_not_globally_optimal_claim=True,all_owned_processes_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,hardware_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,local_rust_compilation=False,prior_capsules_unchanged=prior)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),goal_complete=False)))
