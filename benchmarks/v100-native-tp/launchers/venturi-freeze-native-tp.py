import pathlib,shutil,json,hashlib,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');p=root/'benchmarks/v100-native-tp';src=p/'source';assert not src.exists();src.mkdir()
names=[]
for part in ('src','tests','tools'):
 for f in (root/part).rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and (f.suffix in ('.py','.cu','.cuh','.h','.hpp','.cpp','.c','.json','.md','.sh','.toml') or f.name in ('LICENSE','NOTICE')):
   names.append(f.relative_to(root))
names += [pathlib.Path(x) for x in ['AGENTS.md','README.md','pyproject.toml','docs/native-q8-goal.md','docs/native-mtp-plan.md','scripts/venturi-4090d','scripts/venturi-qwen-4090d','benchmarks/qwen-streaming/case.json','benchmarks/v100-native-tp/summarize.py','benchmarks/v100-native-tp/summary.md']]
hashes={}
for name in sorted(set(names)):
 f=root/name;assert f.is_file() and not f.is_symlink()
 dst=src/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);hashes[str(name)]=hashlib.sha256(dst.read_bytes()).hexdigest()
(p/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
launch=p/'launchers';launch.mkdir()
for name in ['venturi-warp-dispatch.py','venturi-tp-test-dispatch.py','venturi-tp-test-dispatch-v2.py','venturi-tp-full-dispatch.py','venturi-tp-diagnostics-dispatch.py','venturi-tp-diagnostics-dispatch-v2.py','venturi-tp-diagnostics-dispatch-v3.py','venturi-fetch-native-tp.py','venturi-freeze-native-tp.py']:
 shutil.copy2(pathlib.Path('/tmp')/name,launch/name)
remote=p/'v100';metrics=json.loads((p/'metrics.json').read_text())
for job in ['full','diagnostics','validation-v2','warp']:
 assert (remote/job/'exit-code.txt').read_text().strip()=='0'
for job in ['full','diagnostics','validation-v2']:
 proc=json.loads((remote/job/'process.json').read_text())
 for file,sha in proc['source_sha256'].items(): assert hashes[file]==sha,(job,file)
lib=json.loads((remote/'diagnostics/library-integrity.json').read_text());assert lib['sha256']==metrics['library_sha256']==lib['manifest']['library_sha256']
for file,sha in lib['manifest']['source_sha256'].items():assert hashes['src/venturi/ops/csrc/'+file]==sha
for job in ['warp','validation-v1-failed']:
 old=json.loads((remote/job/'source-core-sha256.json').read_text())
 assert all(hashlib.sha256((remote/job/'source-core'/f).read_bytes()).hexdigest()==sha for f,sha in old.items())
for e in json.loads((remote/'warp/libraries.json').read_text()):
 for f,sha in e['manifest']['source_sha256'].items():assert hashlib.sha256((remote/'warp/source-core/src/venturi/ops/csrc'/f).read_bytes()).hexdigest()==sha
prior={}
for name in ['v100-q8-optimization','native-greedy','qwen-streaming','fp32-small-parameters','sm70-fp16']:
 base=root/'benchmarks'/name;old=json.loads((base/'source-sha256.json').read_text())
 assert all(hashlib.sha256((base/'source'/f).read_bytes()).hexdigest()==sha for f,sha in old.items())
 evidence=base/'evidence-sha256.json'
 if evidence.exists():
  data=json.loads(evidence.read_text());assert all(hashlib.sha256((base/f).read_bytes()).hexdigest()==sha for f,sha in data.items())
 prior[name]=dict(source_files=len(old),all_source_hashes_match=True,evidence_manifest_verified=evidence.exists())
cli=json.loads((remote/'diagnostics/cli-smoke.json').read_text());assert cli['generated_ids']==[57590,248046] and cli['native_memory']['position']==27
assert cli['native_memory']['peer_transport']=='host'
assert not metrics['native_mtp_implemented'] and metrics['exact_trained_logits_hidden_gdn_kv']
probe=metrics['peer_probe']['directions'];assert all(d['advertised_peer_access'] for d in probe)
assert [d['direct_copy_bytes_equal'] for d in probe]==[False,False]
assert all(d['explicit_host_copy_bytes_equal'] for d in probe)
assert (remote/'diagnostics/hardware-counter-policy.txt').read_text().strip().endswith('1')
checks=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(hashes),executed_critical_sources_match_frozen_source=True,
 library_integrity_verified=True,full_trained_logits_hidden_gdn_kv_byte_parity=True,fixed_cpu_reference_logit_gates_unchanged=True,
 exact_ABBA_and_story_tokens=True,no_decode_weight_uploads=True,native_CLI_passed=True,final_workers_terminal_success=True,
 local_tests=dict(passed=714,skipped=468),v100_full_tests=dict(passed=769,skipped=413),v100_selected_tp_tests=dict(passed=126,skipped=43),
 experimental_warp_tests=dict(each_passed=53,each_skipped=51,variants=2,promoted=False),
 failed_initial_tp_worker_retained=True,failed_diagnostic_collection_and_incomplete_deployment_retained=True,initial_incomplete_freeze_probe_assumption_corrected_and_preserved=True,
 numpy_dependency='1.26.4 installed only into isolated job overlay, not shared venv',
 bidirectional_peer_capability_advertised=True,both_final_direct_copy_directions_failed=True,explicit_pinned_host_copies_exact=True,
 original_mtp_inventory_only_not_implemented=True,achieved_hardware_counters_available=False,driver_policy_changed=False,
 SM89_exercised_this_milestone=False,parallel_request_serving_qualified=False,matched_external_engine_comparison=False,
 service_state='Previously authorized large llama-server container left stopped; smaller services not stopped this milestone',
 prior_capsules=prior,goal_complete=False,committed_or_pushed_this_milestone=False)
(p/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and f.name!='evidence-sha256.json'}
(p/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2)+'\n')
assert all(hashlib.sha256((p/f).read_bytes()).hexdigest()==sha for f,sha in evidence.items())
print(json.dumps(dict(source_files=len(hashes),frozen_files=len(evidence),checks='passed',single_decode_tps=metrics['single']['decode']['output_tokens_per_wall_second'],parallel_decode_tps=metrics['parallel']['decode']['output_tokens_per_wall_second']),indent=2))
