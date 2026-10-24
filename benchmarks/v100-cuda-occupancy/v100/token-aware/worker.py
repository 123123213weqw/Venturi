import os,pathlib,subprocess,json,hashlib,time,traceback
ROOT=pathlib.Path('/home/wzu/codex-build/Venturi');SRC=ROOT/'v100-occupancy-src-v1';OUT=ROOT/'v100-occupancy-prefill-v2'
PY='/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
BASE=pathlib.Path((ROOT/'v100-pp-final-v1/library.txt').read_text().strip());old=json.loads((BASE.parent/'build.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(ROOT/'v100-tp-diagnostics-v2/python-deps')+':'+str(SRC/'src')+':'+str(SRC/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',VENTURI_PP_TEST='1',VENTURI_TP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1',VENTURI_Q8_BASELINE_LIBRARY=str(BASE))
def run(cmd,log,timeout=900):
 with (OUT/log).open('w') as f:subprocess.run(cmd,cwd=SRC,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 assert json.loads((ROOT/'v100-occupancy-variants-v1/completed.json').read_text())['completed']
 prior=ROOT/'v100-occupancy-prefill-v1'
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
 run([PY,'-m','pytest','-q','--tb=short'],'tests.log',timeout=600)
 for phase in range(4):
  selected=BASE if phase in (0,3) else library
  for mode in ('tp','pp'):
   run([PY,'-u','tools/benchmark_qwen_native_parallel.py',str(ROOT/'models/qwen3.5-27b-text-q8.ninfer'),'--library',str(selected),'--case',str(ROOT/'v100-pp-final-v1/case.json'),'--mode',mode,'--output-directory',str(OUT/f'phase-{phase}-{mode}'),'--concurrency','1,8','--repeats','1','--new-tokens','32','--split-layer','32'],f'phase-{phase}-{mode}.log')
 run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-final.txt')
 (OUT/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:
 (OUT/'failure.txt').write_text(traceback.format_exc());rc=1
(OUT/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
