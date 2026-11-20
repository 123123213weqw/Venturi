import pathlib,json,hashlib,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-q8-wmma';raw=out/'v100'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'evidence-sha256.json').exists(),'Already frozen'
terminal=read(out/'remote-terminal-audit.json')
for name in ['fixtures-v1','projection-v2','t8-v3','guarded-v4','campaign-v5','final-v6']:
 assert int((raw/name/'exit-code.txt').read_text())==0,name
 assert terminal['workers'][name]['exit_code']==0 and terminal['workers'][name]['remaining_process_state'] is None
assert '1206 passed, 440 skipped' in (raw/'final-v6/tests.log').read_text()
assert '761 passed, 885 skipped' in (out/'local-tests.log').read_text()
assert '242 passed' in (raw/'guarded-v4/profile-tests.log').read_text()
assert '130 passed, 125 skipped' in (raw/'guarded-v4/default-regression-tests.log').read_text()
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
q=read(raw/'guarded-v4/qualification/report.json');assert q['completed'] and not q['broad_bf16_corpus_quality_qualified']
for controls in q['profiles'].values():
 assert controls['same_profile_bytes_equal'] and controls['control']==controls['candidate']
 for mode in ['control','candidate']:
  p=controls[mode];assert p['passed'] and len(p['cases'])==4
  assert sum(len(c['all_logit_bytes']) for c in p['cases'])==98
  assert all(c['native_generate_without_torch_math'] and c['diagnostic_generate_state_bytes_equal'] for c in p['cases'])
  assert len(p['batch']['private_cache'])==8 and len(p['reset_continuation']['private_cache'])==8
assert q['prior_report_sha256']==sha(root/'benchmarks/v100-shared-inputs/v100/qualification-v2/qualification/report.json')
campaign=read(raw/'campaign-v5/campaign-report.json');assert campaign['completed'] and campaign['promoted'] and len(campaign['experiments'])==8 and not campaign['tensor_core_model_dispatch']
for e in campaign['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v5'/e['label']/'timed/report.json');assert r['completed']
 counts=[1,8] if e['label'].startswith('screen') else ([1,2,4,8] if e['label'] in ['repeat-control','repeat-t8','repeat-llama-tensor-off'] else [1])
 assert [s['request_count'] for s in e['summary']]==counts
 assert all(s['trials']==(1 if e['label'].startswith('screen') else 3) for s in e['summary'])
 assert len(r['waves'])==len(counts)*(1 if e['label'].startswith('screen') else 3)
 if e['engine']=='venturi':
  wanted='venturi-shared-input-fixtures-v1/build/77fedac214853798/libventuri_blocks.so' if e['label'].endswith('control') else 'venturi-wmma-guarded-v4/build/8537a2e208c36c23/libventuri_blocks.so'
  assert wanted in (raw/'campaign-v5'/e['label']/'loaded-maps.txt').read_text()
 assert all(w['emitted_tokens']==32*w['request_count'] and all(x['response']['tokens']==r['case']['expected_generated_ids'] for x in w['requests']) for w in r['waves'])
assert campaign['reused_trained_mtp_state_reference']['sha256']==sha(root/'benchmarks/v100-shared-inputs/v100/campaign-v3/mtp-qualification/report.json')
assert campaign['qualification_report_sha256']==sha(raw/'guarded-v4/qualification/report.json')
proof=read(raw/'final-v6/executed-source.json')
for rel,h in proof.items():assert sha(root/rel)==h,rel
assert campaign['source_sha256']==terminal['executed_snapshots']['wmma-src-v5']
assert not terminal['owned_inference_processes'] and not terminal['listening_port_18949']
assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy']
assert '3651 MiB' in terminal['gpu_state'] and '3189 MiB' in terminal['gpu_state']
snapshot_matches={}
for snap,files in terminal['executed_snapshots'].items():
 delta=[]
 for rel,h in files.items():
  if (root/rel).is_file() and sha(root/rel)==h:continue
  p=out/'executed-source-deltas'/snap/rel;assert p.is_file() and sha(p)==h,(snap,rel);delta.append(rel)
 snapshot_matches[snap]=dict(total=len(files),preserved_executed_deltas=delta)
builds={}
for directory in ['fixtures-v1','t8-v3','guarded-v4']:
 b=read(next((raw/directory/'build').rglob('build.json')));assert b['architecture']=='sm70' and len(b['source_sha256'])==7
 snap={'fixtures-v1':'wmma-src-v1','t8-v3':'wmma-src-v3','guarded-v4':'wmma-src-v4'}[directory]
 for rel,h in b['source_sha256'].items():
  p=root/'src/venturi/ops/csrc'/rel
  if sha(p)!=h:
   choices=[out/'executed-source-deltas'/snap/'src/venturi/ops/csrc'/rel,out/'experiments'/('dp4a-t8-all' if directory=='t8-v3' else 'dp4a-t4')/rel]
   assert any(v.is_file() and sha(v)==h for v in choices),(directory,rel)
 builds[directory]=b['library_sha256']
assert builds['guarded-v4']==campaign['candidate_library_sha256']==q['candidate_library_sha256']
assert campaign['baseline_library_sha256']==q['baseline_library_sha256']
for directory in ['guarded-v4','final-v6']:assert read(raw/directory/'completed.json')['sha256']==builds['guarded-v4']
for rel,h in terminal['native_source_sha256'].items():assert sha(root/'src/venturi/ops/csrc'/rel)==h
resources=read(raw/'final-v6/resources.json');assert resources['library_sha256']==builds['guarded-v4'] and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
for c in [1,8]:
 m=read(raw/'final-v6'/f'q8-residual-c{c}-metadata.json');assert m['profiled'] and m['logical_requests']==c
 a=read(raw/'final-v6'/f'q8-residual-c{c}-attribution.json');assert len(a['cohorts'])==1 and a['cohorts'][0]['requests']==1
 assert not any('vwt::' in k['name'] for k in a['cohorts'][0]['kernel_by_name'])
for name in ['qwen_runtime.cuh','qwen_batch.cuh','qwen_pipeline.cuh','qwen_mtp.cuh']:
 assert sha(root/'src/venturi/ops/csrc'/name)==sha(root/'benchmarks/v100-shared-inputs/source/src/venturi/ops/csrc'/name)
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
assert len(prior)==21,len(prior)
shutil.copy2(__file__,out/'launchers/freeze.py')
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools',str(out/'summarize.py')],cwd=root,stdout=(out/'ruff.log').open('w'),check=True)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
metrics=read(out/'metrics.json');assert not metrics['external_performance_goal_complete']
assert all(v<1 for v in metrics['t8_vs_fresh_llama_ratio'].values())
assert all(v>1 for v in metrics['t8_vs_control_speedup'].values())
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 if (root/name).exists():files.append(root/name)
files += [out/'summary.md',out/'summarize.py'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_current_source_files_verified=len(proof),executed_snapshot_matches=snapshot_matches,native_libraries_sha256=builds,native_seven_build_sources_verified=True,default_fp16_arithmetic_and_fixed_byte_gates_preserved=True,guarded_dp4a_schedule_byte_exact=True,wmma_research_only_rejected_for_inference=True,trained_fixed_cpu_and_original_trajectory_gates_passed=True,broad_original_bf16_corpus_quality_qualified=False,same_profile_scalar_batch_state_exact=True,trained_mtp_state_reference_explicitly_reused=True,prior_reference_pins_verified=True,three_repeat_promoted_matrix_complete=True,all_measured_external_cells_still_lose=True,all_owned_workers_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,hardware_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,local_rust_compilation=False,prior_capsules_unchanged=prior)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),goal_complete=False)))
