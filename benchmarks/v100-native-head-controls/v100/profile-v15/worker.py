PROFILE_SCRIPT='"""Separate native TP scalar/cohort timeline; never throughput evidence.\n\nBatch has one native_unsplit C-call range for eight logical requests. Scalar has\neight native_unsplit submitting-thread ranges. Do not equate range/request count.\n"""\n\nimport argparse\nfrom concurrent.futures import ThreadPoolExecutor\nimport json\nfrom pathlib import Path\n\nimport torch\n\nfrom benchmark_qwen_native_parallel import run_wave\nfrom venturi.models import NativeQwenModel, QwenQ8Weights\nfrom venturi.ninfer import NInferArtifact\nfrom venturi.ops import CudaBackend\n\n\ndef main():\n    p = argparse.ArgumentParser(description=__doc__)\n    p.add_argument("artifact")\n    p.add_argument("--library", required=True)\n    p.add_argument(\n        "--projection-rows",\n        type=int,\n        choices=(32, 64, 128, 256),\n        default=32,\n        help="Internal projection workspace rows; not per-slot context capacity",\n    )\n    p.add_argument("--case", required=True)\n    p.add_argument("--mode", choices=("scalar", "batch"), required=True)\n    p.add_argument(\n        "--projection-profile",\n        choices=("fp16", "q8-residual", "q8-residual-column-mlp", "q8-residual-column-mlp-head-gdn"),\n        default="fp16",\n    )\n    p.add_argument("--count", type=int, default=8, choices=(1, 2, 4, 8))\n    p.add_argument("--output", required=True)\n    a = p.parse_args()\n    torch.set_num_threads(4)\n    case = json.loads(Path(a.case).read_text())\n    prompt, expected = case["prompt_token_ids"], case["expected_generated_ids"]\n    model = NativeQwenModel.from_q8(\n        QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact)),\n        CudaBackend(a.library),\n        weight_mode="resident",\n        dtype=torch.float16,\n        device="cuda:0",\n        max_tokens=a.projection_rows,\n        tensor_parallel_device="cuda:1",\n        tensor_parallel_transport="host",\n        projection_profile=a.projection_profile,\n    )\n    sessions, batch = [], None\n    try:\n        if a.mode == "batch":\n            batch = model.create_batch([256] * 8)\n            assert (\n                batch.generate([prompt] * a.count, max_new_tokens=len(expected))\n                == [expected] * a.count\n            )\n            batch.reset()\n            with torch.cuda.nvtx.range("native_unsplit"):\n                actual = batch.generate(\n                    [prompt] * a.count, max_new_tokens=len(expected)\n                )\n            assert actual == [expected] * a.count\n            result = dict(batch_info=batch.info(), native_call_ranges=1)\n        else:\n            sessions = [model.create_session(256) for _ in range(8)]\n            with ThreadPoolExecutor(max_workers=a.count) as pool:\n                calls = [\n                    lambda s=s: s.generate(prompt, max_new_tokens=len(expected))\n                    for s in sessions[: a.count]\n                ]\n                warm = run_wave(pool, calls, len(expected))\n                assert all(r["ids"] == expected for r in warm["requests"])\n                for s in sessions:\n                    s.reset()\n                measured = run_wave(pool, calls, len(expected), "native_unsplit")\n                assert all(r["ids"] == expected for r in measured["requests"])\n            result = dict(native_call_ranges=a.count, instrumented_wave=measured)\n        Path(a.output).write_text(\n            json.dumps(\n                dict(\n                    scope=__doc__,\n                    profiled=True,\n                    logical_requests=a.count,\n                    projection_profile=a.projection_profile,\n                    projection_rows=a.projection_rows,\n                    mode=a.mode,\n                    **result,\n                ),\n                indent=2,\n            )\n            + "\\n"\n        )\n    finally:\n        if batch is not None:\n            batch.close()\n        for s in sessions:\n            s.close()\n        model.close()\n\n\nif __name__ == "__main__":\n    main()\n'
import pathlib,subprocess,os,time,json,hashlib,signal,traceback
root=pathlib.Path('/home/data/wangyue');src=root/'repos/codex-build/Venturi/native-head-gdn-src-v3';out=root/'artifacts/venturi-native-head-controls-profile-v15';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests')+':'+str(src/'tools'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,name,timeout=600):
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/name).open('wb') as log:
  p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except BaseException:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned command exited{rc}: {name}')
def identity(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
try:
 campaign=root/'artifacts/venturi-native-head-controls-v13';aligned=root/'artifacts/venturi-native-head-aligned-v14'
 for prior in [campaign,aligned]:
  deadline=time.monotonic()+3600
  while not (prior/'exit-code.txt').exists():
   pid=json.loads((prior/'process.json').read_text())['pid'];cmd=pathlib.Path(f'/proc/{pid}/cmdline')
   assert cmd.exists() and str(prior/'worker.py').encode() in cmd.read_bytes(),'Prerequisite handle missing; inspect rather than replay'
   assert time.monotonic()<deadline,'Finite wait expired; inspect before any continuation'
   time.sleep(15)
  assert (prior/'exit-code.txt').read_text().strip() in (['0'] if prior==campaign else ['0','1']),'Controls must pass; independent diagnostic must be terminal to avoid GPU overlap'
 report=json.loads((campaign/'campaign-report.json').read_text());assert report['completed'] and report['path_promotions']['ordinary256']
 actual={str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==report['source_sha256'];(out/'executed-source.json').write_text(json.dumps(actual,indent=2)+'\n')
 library=pathlib.Path((campaign/'library.txt').read_text().strip());assert sha(library)=='33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8';(out/'library.txt').write_text(str(library)+'\n')
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json'
 before={str(p):identity(p) for p in [target,draft]};previous=json.loads((campaign/'payload-after.json').read_text());assert all(previous[k]==v for k,v in before.items());(out/'payload-before.json').write_text(json.dumps(before,indent=2)+'\n')
 script=out/'profile_ordinary256.py';script.write_text(PROFILE_SCRIPT)
 original=(src/'tools/profile_qwen_batch.py').read_text();expected=original.replace('choices=("fp16", "q8-residual", "q8-residual-column-mlp"),','choices=("fp16", "q8-residual", "q8-residual-column-mlp", "q8-residual-column-mlp-head-gdn"),');assert PROFILE_SCRIPT==expected,'Only explicit CLI profile choice may differ; production source untouched'
 traces=[]
 for arm,profile in [('control','q8-residual-column-mlp'),('candidate','q8-residual-column-mlp-head-gdn')]:
  name=arm+'-ordinary256-c8'
  cmd=[py,str(script),str(target),'--library',str(library),'--projection-rows','256','--case',str(case),'--mode','batch','--count','8','--projection-profile',profile,'--output',str(out/(name+'-metadata.json'))]
  run(['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--cuda-event-trace=false','--cuda-graph-trace=node','-o',str(out/name)]+cmd,name+'-profile.log',timeout=1200)
  run(['/usr/local/bin/nsys','export','--type=sqlite','--output',str(out/(name+'.sqlite')),str(out/(name+'.nsys-rep'))],name+'-export.log',timeout=300)
  run([py,'tools/summarize_native_parallel_trace.py',str(out/(name+'.sqlite')),'--output',str(out/(name+'-attribution.json'))],name+'-attribution.log',timeout=300)
  traces += ['--trace',name+'='+str(out/(name+'.sqlite'))]
 run([py,'tools/inspect_cuda_occupancy.py','--library',str(library),'--expected-sha256',sha(library),'--output',str(out/'resources.json')]+traces,'resources.log',timeout=600)
 after={str(p):identity(p) for p in [target,draft]};assert before==after;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+'\n')
 retained={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']};(out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library_sha256=sha(library),separate_from_unprofiled=True,achieved_hardware_counters=None,production_source_unchanged=True,profile_script_difference='CLI profile choice only',ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
