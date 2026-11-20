from pathlib import Path
import subprocess,json,hashlib,shlex,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
root=Path.cwd();out=root/'benchmarks/v100-dp2a-pilot';remote='/home/data/wangyue/repos/codex-build/Venturi/dp2a-pilot-src-v2'
r=subprocess.run(ns['SSH'],input=f"import pathlib\np=pathlib.Path({remote!r});p.mkdir()\nprint(p)\n",text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
ssh=ns['SSH'][:-2];cmd=['rsync','-az','-e',shlex.join(ssh)]
for p in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',p]
cmd+=['src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/']
for attempt in range(2):
 r=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
 with (out/'sync-v2.log').open('a') as f:f.write(str(attempt)+'\n'+r.stdout+r.stderr)
 if r.returncode==0:break
 if attempt==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(out/'source-manifest-v2.json').write_text(json.dumps(expected,indent=2)+'\n')
source=r'''import pathlib,subprocess,os,json,time,hashlib,traceback,signal
root=pathlib.Path('/home/data/wangyue');src=pathlib.Path(REMOTE);out=root/'artifacts/venturi-dp2a-pilot-campaign-v2';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,name,timeout=300):
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/name).open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned command exited {rc}: {name}')
try:
 expected=EXPECTED
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 assert actual==expected,'Full source mismatch before execution'
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+'\n')
 gate=root/'artifacts/venturi-dp2a-pilot-fixtures-v1';assert (gate/'exit-code.txt').read_text().strip()=='0'
 build=json.loads((gate/'build.json').read_text());lib=pathlib.Path(build['library']);assert sha(lib)==build['library_sha256']
 for r,h in build['source_sha256'].items():assert sha(src/r)==h
 assert sha(src/'tests/test_q8_dp2a.py')==json.loads((gate/'executed-source.json').read_text())['tests/test_q8_dp2a.py']
 (out/'reused-fixtures.json').write_text(json.dumps(dict(path=str(gate),test_log_sha256=sha(gate/'tests.log'),library_sha256=sha(lib),same_test_and_native_source=True),indent=2)+'\n')
 control=root/'artifacts/venturi-native-column-mlp-fixtures-v1/build/77588f610f53223e/libventuri_blocks.so';assert sha(control)=='19ee200feef03d2ddd10e313b680c393fbf8ee170f367d08d0bbd6b5edaebc13'
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'
 cmd=[py,'tools/benchmark_q8_dp2a.py',str(target),'--library',str(lib),'--baseline-library',str(control)]
 run([py,'-m','pytest','-q','tests/test_cuda_occupancy_tool.py','--tb=short'],'occupancy-tool-tests.log')
 run(cmd+['--output-directory',str(out/'timed')],'benchmark.log',timeout=600)
 report=json.loads((out/'timed/report.json').read_text());assert report['completed'] and len(report['cases'])==18
 # All direct profiling follows completed unprofiled trials, never overlaps.
 run(['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--cuda-event-trace=false','--cuda-graph-trace=node','-o',str(out/'projection')]+cmd+['--profile','--output-directory',str(out/'profile')],'profile.log',timeout=300)
 run(['/usr/local/bin/nsys','export','--type=sqlite','--output',str(out/'projection.sqlite'),str(out/'projection.nsys-rep')],'export.log')
 run([py,'tools/summarize_q8_projection_trace.py',str(out/'projection.sqlite'),'--output',str(out/'attribution.json')],'attribution.log')
 for label,library in [('pilot',lib),('control',control)]:
  run([py,'tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',sha(library),'--trace','projection='+str(out/'projection.sqlite'),'--output',str(out/(label+'-resources.json'))],label+'-resources.log')
 sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(lib)],text=True,capture_output=True,check=True).stdout
 (out/'pilot-sass.txt').write_text(sass)
 (out/'pilot-sass-integer-evidence.json').write_text(json.dumps(dict(sass_sha256=sha(out/'pilot-sass.txt'),idp2a_instructions=sass.count('IDP.2A'),idp4a_instructions=sass.count('IDP.4A'),sample_integer_instructions=[l.strip() for l in sass.splitlines() if 'IDP.' in l][:12],scope='Static instruction text, not executed instruction counters or throughput'),indent=2)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
 retained={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [out/'projection.sqlite',out/'projection.nsys-rep',out/'pilot-sass.txt']}
 (out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,ended_unix=time.time(),promote_to_model=report['promote_to_model'],median_fused_operator_ratio=report['median_fused_operator_ratio'],model_benchmark_launched=False,full_goal_complete=False),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''.replace('REMOTE',repr(remote)).replace('expected=EXPECTED','expected='+repr(expected))
ns['dispatch']('venturi-dp2a-pilot-campaign-v2',source)
