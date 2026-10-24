import pathlib,json,hashlib,shutil,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');p=root/'benchmarks/v100-native-pp';remote=p/'v100'
metrics=json.loads((p/'metrics.json').read_text())
for job in ['final','diagnostics','initial-comparison','initial-qualification','validation-v1','validation-v2']:
 assert (remote/job/'exit-code.txt').read_text().strip()=='0',job
src=p/'source';assert not src.exists();src.mkdir()
names=[]
for part in ['src','tests','tools']:
 for f in (root/part).rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and (f.suffix in ('.py','.cu','.cuh','.h','.hpp','.cpp','.c','.json','.md','.sh','.toml') or f.name in ('LICENSE','NOTICE')):names.append(f.relative_to(root))
names += [pathlib.Path(n) for n in ['AGENTS.md','README.md','pyproject.toml','docs/native-q8-goal.md','docs/native-mtp-plan.md','scripts/venturi-4090d','scripts/venturi-qwen-4090d','benchmarks/v100-native-pp/summarize.py','benchmarks/v100-native-pp/summary.md']]
hashes={}
for name in sorted(set(names)):
 f=root/name;assert f.is_file() and not f.is_symlink();dst=src/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);hashes[str(name)]=hashlib.sha256(dst.read_bytes()).hexdigest()
(p/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
for job in ['final','diagnostics']:
 for f,h in json.loads((remote/job/'process.json').read_text())['source_sha256'].items():assert hashes[f]==h,(job,f)
libs=json.loads((remote/'diagnostics/library-integrity.json').read_text());assert libs['final']['sha256']==metrics['library_sha256']
for n,lib in libs.items():
 assert lib['sha256']==lib['manifest']['library_sha256']
 for f,h in lib['manifest']['source_sha256'].items():
  q=(src/'src/venturi/ops/csrc'/f) if n=='final' else (remote/'initial-comparison/source-core/src/venturi/ops/csrc'/f)
  assert hashlib.sha256(q.read_bytes()).hexdigest()==h,(n,f)
for job in ['validation-v1','validation-v2','initial-comparison']:
 old=json.loads((remote/job/'source-core-sha256.json').read_text());assert all(hashlib.sha256((remote/job/'source-core'/f).read_bytes()).hexdigest()==h for f,h in old.items())
for job in ['final','initial-qualification']:
 r=json.loads((remote/job/'qualification/report.json').read_text());assert r['native_full_runtime_case_passed'] and r['all_native_logit_hidden_state_bytes_match_previous'] and r['all_fixed_cpu_logit_gates_passed'] and r['native_cpp_generation_without_torch_math_completed']
cli=json.loads((remote/'diagnostics/cli-smoke.json').read_text());assert cli['generated_ids']==[57590,248046] and cli['native_memory']['position']==27
assert cli['native_memory']['parallelism']=='pipeline' and cli['native_memory']['pipeline_split_layer']==32
assert 'stage' in cli['scheduling'].lower()
assert metrics['final_stage_selection_atomic'] and all(c['tp']['trials']==2 and c['pp']['trials']==2 for c in metrics['counts'].values())
for key,r in metrics['profiler_only'].items():
 count=int(key.split('-c')[1]);assert len(r['cohorts'])==3 and all(c['requests']==count for c in r['cohorts'])
assert json.loads((remote/'failed-profiler-event-trace/abort-record.json').read_text())['report_completed'] is False
assert not json.loads((remote/'failed-profiler-event-trace/tp-c8/profiled-run/report.json').read_text()).get('completed',False)
assert '--cuda-event-trace=false' in (remote/'diagnostics/worker.sh').read_text()
assert (remote/'diagnostics/hardware-counter-policy.txt').read_text().strip().endswith('1')
prior={}
for name in ['v100-native-tp','v100-q8-optimization','native-greedy','qwen-streaming','fp32-small-parameters','sm70-fp16']:
 base=root/'benchmarks'/name;old=json.loads((base/'source-sha256.json').read_text());assert all(hashlib.sha256((base/'source'/f).read_bytes()).hexdigest()==h for f,h in old.items())
 e=base/'evidence-sha256.json'
 if e.exists():
  d=json.loads(e.read_text());assert all(hashlib.sha256((base/f).read_bytes()).hexdigest()==h for f,h in d.items())
 prior[name]=dict(source_files=len(old),source_and_available_evidence_manifests_verified=True)
launch=p/'launchers';launch.mkdir()
for name in ['venturi-pp-edit-core.py','venturi-pp-edit-python.py','venturi-pp-dispatch.py','venturi-pp-dispatch-v2.py','venturi-pp-full-dispatch.py','venturi-pp-benchmark-dispatch.py','venturi-pp-profile-dispatch.py','venturi-pp-final-dispatch.py','venturi-pp-profile-dispatch-v2.py','venturi-pp-profile-dispatch-v3.py','venturi-fetch-native-pp.py','venturi-freeze-native-pp.py']:
 shutil.copy2(pathlib.Path('/tmp')/name,launch/name)
checks=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(hashes),executed_final_critical_sources_match_frozen_source=True,
 initial_and_final_library_manifests_verified=True,initial_source_counterexample_and_timing_preserved=True,exact_trained_logits_hidden_gdn_history_kv=True,
 fixed_cpu_reference_gates_unchanged=True,full_native_generation_with_torch_math_forbidden=True,concurrent_fixture_distinct_prompt_state_isolation_at=[1,2,4,8],
 final_unprofiled_correctness_first_ABBA_all_ids_match=True,global_wall_throughput_not_summed_latencies=True,
 local_tests=dict(passed=721,skipped=494),v100_full_tests=dict(passed=789,skipped=426),final_native_fixture_tests=dict(passed=30,skipped=27),
 final_workers_terminal_success=True,superseded_profiler_waiter_stopped_before_capture=True,failed_device_event_trace_capture_retained=True,final_traces_disable_profiler_device_event_completion=True,
 native_pipeline_cli_passed=True,hardware_counters_available=False,driver_policy_changed=False,SM89_exercised=False,
 checkpoint_scope='Original-derived Q8 group32 FP16 represented profile, not independent original-BF16/corpus/upstream quality',
 performance_scope='Native concurrent sessions, existing serialized TP versus overlapping PP stages; not HTTP/continuous batching/paged KV/MTP or matched external engine',
 no_other_gpu_services_stopped=True,prior_capsules=prior,goal_complete=False,committed_or_pushed=False)
(p/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n')
evidence={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and f.name!='evidence-sha256.json'}
(p/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2)+'\n');assert all(hashlib.sha256((p/f).read_bytes()).hexdigest()==h for f,h in evidence.items())
print(json.dumps(dict(source_files=len(hashes),frozen_evidence_files=len(evidence),all_checks='passed',counts={n:dict(tp=c['tp']['decode']['aggregate_output_tokens_per_wall_second'],pp=c['pp']['decode']['aggregate_output_tokens_per_wall_second']) for n,c in metrics['counts'].items()}),indent=2))
