import pathlib,json,hashlib,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-q8-dp4a';raw=out/'v100'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'evidence-sha256.json').exists(),'Already frozen'
for name,code in [('t8-screen-v1',0),('scalar-fixtures-v1',0),('scalar-projection-v2',0),('vector-v3',0),('worker-screen-v1',0),('llama-arithmetic-v1',1),('llama-arithmetic-v2',0),('final-v4',0)]:assert int((raw/name/'exit-code.txt').read_text())==code,name
assert '995 passed, 440 skipped' in (raw/'final-v4/tests.log').read_text()
assert '758 passed, 677 skipped' in (out/'local-tests.log').read_text()
assert '72 passed' in (raw/'scalar-fixtures-v1/dp4a-tests.log').read_text() and '72 passed' in (raw/'vector-v3/dp4a-tests.log').read_text()
for name in ['t8-screen-v1','worker-screen-v1']:assert '130 passed, 125 skipped' in (raw/name/'fixture-tests.log').read_text()
for name in ['scalar-projection-v2','vector-v3']:
 r=read(raw/name/'projection/report.json');assert r['completed'] and len(r['cases'])==32 and not r['full_model_quality_qualified']
 assert all(len(c['waves'])==9 and all(v['relative_l2']<0.001 for v in c['sampled_reference']) for c in r['cases'])
cpp=root/'src/venturi/ops/csrc';previous=root/'benchmarks/v100-native-batch/source'
assert (cpp/'qwen_runtime.cuh').read_bytes()==(previous/'src/venturi/ops/csrc/qwen_runtime.cuh').read_bytes()
assert (cpp/'blocks.cu').read_text().replace('#include "q8_dp4a.cuh"\n','')==(previous/'src/venturi/ops/csrc/blocks.cu').read_text()
assert 'tp_dispatch' not in (root/'src/venturi/ops/build.py').read_text()
for name in ['qwen_native.py','qwen_batch.py','qwen_mtp_native.py','qwen_pipeline.py']:assert sha(root/'src/venturi/models'/name)==sha(previous/'src/venturi/models'/name)
proof=read(raw/'final-v4/executed-source.json')
for rel,h in proof.items():assert sha(root/rel)==h,rel
builds=list(raw.rglob('build.json'));assert len(builds)==5,len(builds)
build_sources={}
for path in builds:
 b=read(path);checks={}
 for name,h in b['source_sha256'].items():
  candidates=[cpp/name,previous/'src/venturi/ops/csrc'/name]+list((out/'experiments').glob('*/'+name))
  matches=[str(p.relative_to(root)) for p in candidates if p.is_file() and sha(p)==h];assert matches,(str(path),name,h);checks[name]=matches
 build_sources[str(path.relative_to(out))]=checks
final=read(raw/'final-v4/completed.json');assert final['completed'];vector=read(raw/'vector-v3/completed.json');assert vector['sha256']==final['sha256']
profiles=read(raw/'final-v4/projection-attribution.json');assert len(profiles['ranges'])==6
for r in profiles['ranges']:assert r['launches']==(16 if r['label'].startswith('projection_0_') else 32)
audit=read(raw/'llama-arithmetic-v2/campaign-report.json');assert audit['completed'] and audit['profiled'] and audit['actual_activation_quantization_kernels'] and audit['actual_quantized_projection_kernels']
assert not read(raw/'llama-arithmetic-v1/observation-recovery.json')['completed']
terminal=read(out/'remote-terminal-audit.json');assert not terminal['owned_inference_processes'];assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy'];assert '3651 MiB' in terminal['gpu_state'] and '3189 MiB' in terminal['gpu_state']
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
assert len(prior)==18
shutil.copy2('/tmp/venturi-dp4a-freeze.py',out/'launchers/freeze.py')
subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools','benchmarks/v100-q8-dp4a/summarize.py'],cwd=root,stdout=(out/'ruff.log').open('w'),check=True);subprocess.run(['git','diff','--check'],cwd=root,check=True)
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for name in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 p=root/name
 if p.exists():files.append(p)
files += [out/'summary.md',out/'summarize.py'];manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');subprocess.run(['python3',str(out/'summarize.py')],cwd=root,check=True)
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(manifest),executed_current_source_files_verified=len(proof),native_build_source_matches=build_sources,default_model_runtime_unchanged=True,new_profiles_explicit_not_full_model_qualified=True,default_precision_path_unchanged=True,failed_t8_and_worker_changes_removed=True,prototype_operator_fp64_gates_passed=True,trained_profile_quality_qualified=False,operator_only_speedups_not_inference_tps=True,actual_competitor_activation_packing_trace=True,failed_observer_and_salvage_preserved=True,all_owned_workers_terminal=True,original_gpu_footprint_restored=True,achieved_occupancy=None,hardware_counter_policy_unchanged=True,external_speed_goal_complete=False,no_commit_or_push=True,local_rust_compilation=False,prior_capsules_unchanged=prior)
(out/'evidence-check.json').write_text(json.dumps(checks,indent=2)+'\n');evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');assert all(sha(out/p)==h for p,h in evidence.items());assert all(sha(out/'source'/p)==h for p,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),goal_complete=False)))
