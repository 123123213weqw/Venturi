from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
root=Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-q8-projection-fusion';raw=out/'v100'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
terminal=read(out/'terminal-audit.json');assert not terminal['owned_inference_processes'] and not terminal['listening_ports'];assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy']
for rel,h in read(out/'prior-reference-pins.json').items():assert sha(root/rel)==h,rel
assert len(read(out/'prior-capsule-audit.json'))==25
for cap_name in read(out/'prior-capsule-audit.json'):
 cap=root/'benchmarks'/cap_name
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  path=cap/name
  if not path.exists():continue
  manifest=read(path);manifest=manifest.get('files',manifest);verified=0;missing=[]
  for rel,h in manifest.items():
   p=base/rel
   if not p.exists() and cap_name=='decoder-layer':p=base/Path(rel).name
   if p.is_file():assert sha(p)==h,str(p);verified+=1
   else:missing.append(rel)
  if missing:assert cap_name=='decoder-layer' and verified==4

assert '765 passed, 1192 skipped' in (out/'local-tests.log').read_text()
prior=read(root/'benchmarks/v100-q8-tiled-staging/v100/model-v3/qualification/report.json')
for tag in ['model-v2','model-v3']:
 d=raw/tag;assert (d/'exit-code.txt').read_text().strip()=='0';q=read(d/'qualification/report.json');assert q['completed'] and q['executed_candidate_profiles']==['q8-residual'] and not q['broad_bf16_corpus_quality_qualified']
 assert q['prior_report_sha256']==sha(root/'benchmarks/v100-q8-tiled-staging/v100/model-v3/qualification/report.json')
 p=q['profiles']['q8-residual'];assert p['same_profile_bytes_equal'] and p['candidate']==p['control']==prior['profiles']['q8-residual']['candidate']
 assert len(p['candidate']['cases'])==4 and sum(len(c['all_logit_bytes']) for c in p['candidate']['cases'])==98
 assert len(p['candidate']['batch']['private_cache'])==len(p['candidate']['reset_continuation']['private_cache'])==8
 assert list(q['unexecuted_profile_reference'])==['fp16']
 mtp=read(d/'mtp-qualification/report.json');assert mtp['completed'] and not mtp['block_reference'] and len(mtp['cases'])==4 and all(c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['native_without_python_math'] and c['mtp_stats']['forced_proposals']==0 for c in mtp['cases'])
for tag in ['study-v1','model-v3']:assert '1517 passed, 440 skipped' in (raw/tag/'full-tests.log').read_text()
assert '221 passed' in (raw/'study-v1/profile-tests.log').read_text()
micro=read(raw/'model-v2/micro/report.json');assert micro['completed'] and len(micro['cases'])==18 and all(c['output_and_workspace_bytes_equal'] and len(c['waves'])==9 and max(c['independent_fp64_relative_l2'])<.001 for c in micro['cases'])
c=read(raw/'campaign-v4/campaign-report.json');assert c['completed'] and c['promoted'] and c['promoted_variant']=='fused' and len(c['experiments'])==10
for e in c['experiments']:
 assert e['completed'] and e['stable_outputs'];r=read(raw/'campaign-v4'/e['label']/'timed/report.json');assert r['completed']
 counts=[1,2,8] if e['label'] in ['screen-control','screen-grouped','screen-fused'] else ([1,2,4,8] if e['label'] in ['repeat-control','repeat-fused'] else [1]);repeats=1 if e['label'].startswith('screen') else 3
 assert [s['request_count'] for s in e['summary']]==counts and all(s['trials']==repeats for s in e['summary']);assert len(r['waves'])==len(counts)*repeats
 assert all(w['emitted_tokens']==32*w['request_count'] and all(x['response']['tokens']==r['case']['expected_generated_ids'] for x in w['requests']) for w in r['waves'])
 expected='venturi-tiled-staging-model-v3' if e['label'].endswith('control') else ('venturi-projection-fusion-v1' if e['label'].endswith('grouped') else 'venturi-projection-fusion-model-v3')
 assert expected in (raw/'campaign-v4'/e['label']/'loaded-maps.txt').read_text()
for gate in c['qualification_gates']:
 d=raw/('model-v2' if gate['path'].endswith('model-v2') else 'model-v3');assert gate['qualification_report_sha256']==sha(d/'qualification/report.json') and gate['mtp_report_sha256']==sha(d/'mtp-qualification/report.json')
proof=read(raw/'final-v5/executed-source.json');assert proof==c['source_sha256']==terminal['executed_snapshots']['projection-fusion-src-v3']
resolved={}
for snap,files in terminal['executed_snapshots'].items():
 delta=[]
 for rel,h in files.items():
  if (root/rel).is_file() and sha(root/rel)==h:continue
  p=out/'executed-source-deltas'/snap/rel;assert p.is_file() and sha(p)==h,(snap,rel);delta.append(rel)
 resolved[snap]=dict(total=len(files),preserved_executed_delta=delta)
for tag,snap in [('study-v1','projection-fusion-src-v1'),('model-v3','projection-fusion-src-v3')]:
 b=read(next((raw/tag/'build').rglob('build.json')));assert b['architecture']=='sm70' and len(b['source_sha256'])==9
 for rel,h in b['source_sha256'].items():assert terminal['executed_snapshots'][snap]['src/venturi/ops/csrc/'+rel]==h
 if tag=='model-v3':assert b['library_sha256']==c['candidate_library_sha256']
resources=read(raw/'final-v5/resources.json');assert resources['library_sha256']==c['candidate_library_sha256'] and resources['achieved_occupancy'] is None and not resources['hardware_counter_collection']
for count in [1,8]:
 for label in ['control','grouped','candidate']:
  a=read(raw/'final-v5'/f'{label}-c{count}-attribution.json')['cohorts'][0];assert a['requests']==1
  selected=any('vqf::' in k['name'] for k in a['kernel_by_name']);assert selected==(label=='candidate')
assert read(raw/'final-v5/micro-results/report.json')['completed'];assert len(read(raw/'final-v5/micro-attribution.json')['ranges'])==27
assert (raw/'final-v5/exit-code.txt').read_text().strip()=='0'
for name in ['q8_dp4a.cuh','q8_shared.cuh','qwen_batch.cuh','qwen_mtp.cuh','qwen_pipeline.cuh']:assert sha(root/'src/venturi/ops/csrc'/name)==sha(root/'benchmarks/v100-q8-tiled-staging/source/src/venturi/ops/csrc'/name)
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools',str(out/'summarize.py')],cwd=root,stdout=(out/'ruff.log').open('w'),check=True)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
subprocess.run(['python3',str(out/'summarize.py')],cwd=root,stdout=(out/'summary-recompute.log').open('w'),check=True)
m=read(out/'metrics.json');assert m['candidate_promoted'] and not m['external_performance_goal_complete'] and all(x<1 for x in m['candidate_vs_reused_external_non_spec_ratio'].values())
assert min(m['same_profile_speedup'].values())>1.04 and m['mtp_http_speedup']>1.07
shutil.copy2(__file__,out/'launchers/freeze.py')
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 if (root/name).exists():files.append(root/name)
files += [out/'summary.md',out/'summarize.py',out/'design.md',out/'next-bottlenecks.md'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_snapshots_verified=resolved,library_sha256=c['candidate_library_sha256'],projection_group_mode=2,transport_only_ablation_qualified=True,model_fused_complete_dot_enabled=True,arithmetic_and_fixed_quality_gates_unchanged=True,trained_same_profile_scalar_batch_mtp_state_bytes_exact=True,unaffected_trained_fp16_reference_reused=True,independent_draft_cpu_reference_reused=True,broad_bf16_corpus_quality_qualified=False,three_repeat_promoted_matrix_complete=True,external_controls_explicitly_reused=True,all_required_external_http_cells_still_lose=True,all_owned_processes_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,driver_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,no_local_rust_or_cuda_compilation=True,prior_capsules_unchanged=read(out/'prior-capsule-audit.json'))
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=25,goal_complete=False)))
