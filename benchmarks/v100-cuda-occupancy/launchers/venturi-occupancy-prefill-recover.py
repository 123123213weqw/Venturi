import ast,pathlib,subprocess
p=pathlib.Path('/tmp/venturi-occupancy-prefill-dispatch.py');tree=ast.parse(p.read_text())
worker=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='worker' for t in n.targets))
worker=worker.replace("OUT=ROOT/'v100-occupancy-prefill-v1'","OUT=ROOT/'v100-occupancy-prefill-v2'")
a=worker.index(" csrc=SRC/'src/venturi/ops/csrc'")
b=worker.index(" run([PY,'-m','pytest','-q','--tb=short'],'tests.log'",a)
pre=r''' prior=ROOT/'v100-occupancy-prefill-v1'
 library=prior/'candidate/libventuri_blocks.so'
 manifest=json.loads((library.parent/'build.json').read_text());assert digest(library)==manifest['library_sha256']
 qualifier=json.loads((prior/'qualification/report.json').read_text())
 for key in ('native_cpp_generation_without_torch_math_completed','generation_matches_previous','all_fixed_cpu_logit_gates_passed','all_native_logit_hidden_state_bytes_match_previous','native_full_runtime_case_passed'):assert qualifier[key]
 csrc=SRC/'src/venturi/ops/csrc';assert all(digest(csrc/f)==h for f,h in old['source_sha256'].items())
 assert all((SRC/f'scripts/{f}').exists() for f in ('venturi-4090d','venturi-qwen-4090d'))
 stray=SRC/'test_cuda_occupancy_tool.py'
 if stray.exists():
  assert digest(stray)==digest(SRC/'tests/test_cuda_occupancy_tool.py');stray.unlink()
 (OUT/'prerequisites.json').write_text(json.dumps(dict(recovery='First full-suite attempt stopped before timing: scripts directory omitted from source sync. Synced scripts, rerun entire suite before any throughput runs. No native code, library or numerical gate changes.',library=str(library),library_sha256=digest(library),previous_qualification_report_sha256=digest(prior/'qualification/report.json'),previous_qualification_passed=True,source_sha256={str(p.relative_to(SRC)):digest(p) for d in ('src','tests','tools','scripts') for p in (SRC/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}),indent=2))
 (OUT/'library.txt').write_text(str(library)+'\n')
 env['VENTURI_SM70_LIBRARY']=str(library)
'''
worker=worker[:a]+pre+worker[b:]
script=f'''import pathlib,subprocess,json,time
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-occupancy-prefill-v2');p.mkdir();(p/'worker.py').write_text({worker!r})
with (p/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=script,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
