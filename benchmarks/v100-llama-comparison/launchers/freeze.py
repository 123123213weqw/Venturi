import pathlib,json,hashlib,shutil,subprocess,datetime
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-llama-comparison';raw=out/'v100'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
assert not (out/'evidence-sha256.json').exists(),'Frozen capsules must not be modified'
terminals={}
for name,expected in [('export-exit-code.txt','0'),('pair-build-exit-code.txt','0'),('pair-screen-v1/exit-code.txt','0'),('screen-exit-code.txt','1'),('recovery-v2/screen-exit-code.txt','1'),('unified-recovery-v3/screen-exit-code.txt','1')]:
 value=(raw/name).read_text().strip();assert value==expected,(name,value);terminals[name]=value
assert read(raw/'export-report.json')['completed']
assert len(read(raw/'export-report.json')['inventory']['mtp_tensor_names'])==15
assert not read(raw/'screen-report.json')['completed']
assert not read(raw/'recovery-v2/screen-report.json')['completed']
assert not read(raw/'unified-recovery-v3/screen-report.json')['completed']
failure,=read(raw/'unified-recovery-v3/screen-report.json')['experiments'];assert failure['server_final_returncode']==-11
assert not read(raw/'final-audit.json')['owned_processes']
assert 'RmProfilingAdminOnly: 1' in read(raw/'final-audit.json')['counter_policy']
assert '17 passed' in (out/'local-adapter-tests.log').read_text()
assert '17 passed' in (raw/'recovery-v2/adapter-tests.log').read_text()
assert '62 passed, 61 skipped' in (raw/'pair-screen-v1/tests.log').read_text()
assert read(raw/'pair-screen-v1/projection/report.json')['completed']
assert all(c['exact_baseline_output_bytes'] for c in read(raw/'pair-screen-v1/projection/report.json')['cases'])
assert not read(raw/'pair-screen-v1/report.json')['default_changed']
old=root/'benchmarks/v100-native-mtp'
previous=read(old/'source-sha256.json')
assert all(sha(root/f)==h for f,h in previous.items() if f.startswith('src/')),'Native runtime changed without a new promotion'
qualified=read(old/'v100/build-final/build.json')
assert all(sha(root/'src/venturi/ops/csrc'/f)==h for f,h in qualified['source_sha256'].items())
prior={}
for cap in sorted((root/'benchmarks').iterdir()):
 if cap==out or not cap.is_dir():continue
 rec={}
 for name,base in [('source-sha256.json',cap/'source'),('evidence-sha256.json',cap)]:
  p=cap/name
  if not p.exists():continue
  m=read(p);m=m.get('files',m);checked=[];missing=[]
  for f,h in m.items():
   p=base/f
   if not p.exists() and cap.name=='decoder-layer':p=base/pathlib.Path(f).name
   if p.is_file():assert sha(p)==h,str(p);checked.append(f)
   else:missing.append(f)
  if missing:assert cap.name=='decoder-layer' and len(checked)==4
  rec[name]=dict(verified=len(checked),legacy_unavailable_files=missing)
 if rec:prior[cap.name]=rec
# Preserve the exact old response validator alongside the repaired version.
legacy=out/'v100/adapter-source-history';legacy.mkdir(exist_ok=True)
for relative,dest in [('tools/benchmark_completion_api.py','initial-client.py'),('tools/serve_native_benchmark.py','initial-server.py')]:
 command=['rsync','-az','-e','ssh -o BatchMode=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o ProxyCommand=none -o ConnectTimeout=20','WZU_Server:/home/wzu/codex-build/Venturi/llama-compare-src-v1/'+relative,str(legacy/dest)]
 subprocess.run(command,check=True,timeout=60)
files=[]
for folder in ['src','tests','tools','scripts','docs']:
 files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(s.startswith('.env') for s in p.parts)]
for f in ['AGENTS.md','pyproject.toml','README.md','LICENSE','NOTICE']:
 p=root/f
 if p.exists():files.append(p)
files += [out/'summary.md',out/'summarize.py']
manifest={}
for p in sorted(set(files)):
 rel=p.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=sha(dest)
(out/'source-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
subprocess.run(['python3',str(out/'summarize.py')],check=True,cwd=root)
metrics=read(out/'metrics.json')
assert all(s['all_measured_ids_equal_pinned_native'] for x in metrics['experiments'] for s in x['summary'])
shutil.copy2('/tmp/venturi-freeze-llama-screen.py',out/'launchers/freeze.py')
result=subprocess.run(['/opt/homebrew/bin/ruff','check','src','tests','tools','benchmarks/v100-llama-comparison/summarize.py'],check=True,cwd=root,text=True,capture_output=True)
(out/'ruff.log').write_text(result.stdout+result.stderr)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
proof=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Finite initial comparison and kernel screen only; revised goal NOT complete',all_owned_workers_terminal=True,terminal_exit_codes=terminals,all_experiments_successful=False,incomplete_external_arms_preserved=True,unified_retry_sigsegv_preserved=True,source_files=len(manifest),prior_capsules_unchanged=prior,native_runtime_and_default_unchanged=True,experimental_pair_kernel_promoted=False,source_original_verified=True,exact_cross_engine_arithmetic=False,achieved_hardware_counters_available=False,old_app_goal_record_modified=False,no_owned_gpu_workers_remaining=True,no_commit_or_push=True,no_local_rust_compilation=True)
(out/'evidence-check.json').write_text(json.dumps(proof,indent=2)+'\n')
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'source' not in p.relative_to(out).parts and p.name!='evidence-sha256.json'}
(out/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
assert all(sha(out/f)==h for f,h in evidence.items());assert all(sha(out/'source'/f)==h for f,h in manifest.items())
print(json.dumps(dict(frozen=True,source_files=len(manifest),evidence_files=len(evidence),prior_capsules_verified=len(prior),revised_goal_completed=False)))
