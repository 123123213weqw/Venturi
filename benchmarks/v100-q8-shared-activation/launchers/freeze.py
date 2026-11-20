import pathlib,json,hashlib,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-q8-shared-activation';raw=out/'v100'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'evidence-sha256.json').exists(),'Already frozen'
terminal=read(out/'remote-terminal-audit.json');assert not terminal['owned_inference_processes'] and not terminal['listening_ports'];assert '3651 MiB' in terminal['gpu_state'] and '3189 MiB' in terminal['gpu_state'];assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy']
for n in ['study-v1','model-v2','campaign-v3','final-v4']:assert int((raw/n/'exit-code.txt').read_text())==0
assert '142 passed' in (raw/'study-v1/profile-tests.log').read_text()
assert '1240 passed, 456 skipped' in (raw/'study-v1/full-tests.log').read_text()
assert '1256 passed, 440 skipped' in (raw/'final-v4/tests.log').read_text()
assert '762 passed, 934 skipped' in (out/'local-tests.log').read_text()
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
q=read(raw/'model-v2/qualification/report.json');assert q['completed'] and not q['broad_bf16_corpus_quality_qualified']
for p in q['profiles'].values():
 assert p['same_profile_bytes_equal'] and p['control']==p['candidate']
 for label in ['control','candidate']:
  v=p[label];assert v['passed'] and len(v['cases'])==4 and sum(len(c['all_logit_bytes']) for c in v['cases'])==98
  assert all(c['native_generate_without_torch_math'] and c['diagnostic_generate_state_bytes_equal'] for c in v['cases'])
  assert len(v['batch']['private_cache'])==len(v['reset_continuation']['private_cache'])==8
mtp=read(raw/'model-v2/mtp-qualification/report.json');assert mtp['completed'] and len(mtp['cases'])==4 and not mtp['block_reference']
assert all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['native_without_python_math'] and c['mtp_stats']['forced_proposals']==0 for c in mtp['cases'])
reference=read(out/'independent-draft-reference.json');assert sha(root/reference['reference_path'])==reference['sha256'];assert len(read(root/reference['reference_path'])['block_reference'])==2
c=read(raw/'campaign-v3/campaign-report.json');assert c['completed'] and c['promoted'] and len(c['experiments'])==8
for e in c['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v3'/e['label']/'timed/report.json');assert r['completed']
 counts=[1,2,8] if e['label'] in ['screen-control','screen-shared-t2'] else ([1,2,4,8] if e['label'] in ['repeat-control','repeat-shared-t2'] else [1]);repeats=1 if e['label'].startswith('screen') else 3
 assert [s['request_count'] for s in e['summary']]==counts and all(s['trials']==repeats for s in e['summary']);assert len(r['waves'])==len(counts)*repeats
 assert all(w['emitted_tokens']==32*w['request_count'] and all(x['response']['tokens']==r['case']['expected_generated_ids'] for x in w['requests']) for w in r['waves'])
 assert ('venturi-wmma-guarded-v4' if e['label'].endswith('control') else 'venturi-shared-activation-model-v2') in (raw/'campaign-v3'/e['label']/'loaded-maps.txt').read_text()
assert c['qualification_report_sha256']==sha(raw/'model-v2/qualification/report.json');assert c['fresh_trained_mtp_state_qualification']['sha256']==sha(raw/'model-v2/mtp-qualification/report.json')
assert c['candidate_library_sha256']==q['candidate_library_sha256']==mtp['library_sha256'];assert c['baseline_library_sha256']==q['baseline_library_sha256']
proof=read(raw/'final-v4/executed-source.json');assert proof==c['source_sha256']==terminal['executed_snapshots']['shared-activation-src-v2']
resolved={}
for snap,files in terminal['executed_snapshots'].items():
 delta=[]
 for rel,h in files.items():
  if (root/rel).is_file() and sha(root/rel)==h:continue
  p=out/'executed-source-deltas'/snap/rel;assert p.is_file() and sha(p)==h,(snap,rel);delta.append(rel)
 resolved[snap]=dict(total=len(files),preserved_executed_delta=delta)
for directory in ['study-v1','model-v2']:
 b=read(next((raw/directory/'build').rglob('build.json')));assert b['architecture']=='sm70' and len(b['source_sha256'])==8
 for rel,h in b['source_sha256'].items():
  p=root/'src/venturi/ops/csrc'/rel
  if sha(p)!=h:assert sha(out/'executed-source-deltas/shared-activation-src-v1/src/venturi/ops/csrc'/rel)==h
 if directory=='model-v2':assert b['library_sha256']==c['candidate_library_sha256']
micro=read(raw/'study-v1/projection/report.json');assert micro['completed'] and len(micro['cases'])==15 and len(micro['unsupported_cases'])==1
assert all(v['residual_control_output_bytes_equal'] and len(v['waves'])==18 for v in micro['cases'])
res=read(raw/'final-v4/resources.json');assert res['library_sha256']==c['candidate_library_sha256'] and res['achieved_occupancy'] is None and not res['hardware_counter_collection']
for count in [1,2,8]:
 m=read(raw/'final-v4'/f'candidate-c{count}-metadata.json');assert m['profiled'] and m['logical_requests']==count
 a=read(raw/'final-v4'/f'candidate-c{count}-attribution.json')['cohorts'][0];assert a['requests']==1
 selected=[k for k in a['kernel_by_name'] if 'vds::dot' in k['name']];assert bool(selected)==(count==2)
 assert not any('vwt::' in k['name'] for k in a['kernel_by_name'])
old=root/'benchmarks/v100-strided-output/source'
for name in ['q8_dp4a.cuh','qwen_batch.cuh','qwen_mtp.cuh','qwen_pipeline.cuh']:assert sha(root/'src/venturi/ops/csrc'/name)==sha(old/'src/venturi/ops/csrc'/name)
prior={}
for cap in sorted((root/'benchmarks').iterdir()):
 if cap==out or not cap.is_dir():continue
 record={}
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  p=cap/name
  if not p.exists():continue
  manifest=read(p);manifest=manifest.get('files',manifest);n=0;missing=[]
  for rel,h in manifest.items():
   path=base/rel
   if not path.exists() and cap.name=='decoder-layer':path=base/pathlib.Path(rel).name
   if path.is_file():assert sha(path)==h,str(path);n+=1
   else:missing.append(rel)
  if missing:assert cap.name=='decoder-layer' and n==4
  record[name]=dict(verified=n,legacy_unavailable_files=missing)
 if record:prior[cap.name]=record
assert len(prior)==23,len(prior)
shutil.copy2(__file__,out/'launchers/freeze.py')
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools',str(out/'summarize.py')],cwd=root,stdout=(out/'ruff.log').open('w'),check=True);subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
m=read(out/'metrics.json');assert not m['external_performance_goal_complete'] and m['candidate_promoted'];assert all(x<1 for x in m['candidate_vs_reused_external_non_spec_ratio'].values());assert m['same_profile_speedup']['2']>1.03 and m['mtp_http_speedup']>1.025
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 if (root/name).exists():files.append(root/name)
files += [out/'summary.md',out/'summarize.py'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_snapshots_verified=resolved,library_sha256=c['candidate_library_sha256'],model_t2_only_staging_enabled=True,t1_shared_operator_not_promoted=True,arithmetic_and_fixed_quality_gates_unchanged=True,trained_same_profile_scalar_batch_mtp_state_bytes_exact=True,broad_bf16_corpus_quality_qualified=False,independent_draft_cpu_reference_reused=True,three_repeat_promoted_matrix_complete=True,all_required_external_cells_still_lose=True,external_controls_explicitly_reused=True,all_owned_processes_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,driver_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,no_local_rust_compilation=True,prior_capsules_unchanged=prior)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),goal_complete=False)))
