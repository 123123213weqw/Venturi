from pathlib import Path
import json
b=Path('benchmarks/v100-native-head-controls');s=Path('/tmp/venturi-native-head-controls-v1-worker.py').read_text();source=s[:s.index('\nEXPECTED=')].replace('venturi-native-head-controls-v1','venturi-native-head-controls-v4')
# Keep native env unchanged. Restore the previously qualified dependency route
# only for the competitor server, and record actual loaded dependencies/settings.
source=source.replace("item['command']=cmd;", "server_env=env.copy()\n  if engine=='llama':server_env['LD_LIBRARY_PATH']=str(bindir)+':/usr/local/cuda/lib64:'+str(root/'repos/codex-build/.venvs/b2-v100/lib/python3.10/site-packages/nvidia/nccl/lib')\n  item['command']=cmd;")
source=source.replace('server=subprocess.Popen(cmd,cwd=src,env=env,stdout=f', 'server=subprocess.Popen(cmd,cwd=src,env=server_env,stdout=f')
source=source.replace("  cmd=[py,'-u','tools/benchmark_completion_api.py'", "  if engine=='llama':\n   actual_env=dict(x.split('=',1) for x in pathlib.Path(f'/proc/{server.pid}/environ').read_bytes().decode().split(chr(0)) if '=' in x)\n   selected_env={k:actual_env.get(k) for k in ['GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','CUDA_VISIBLE_DEVICES','NCCL_P2P_DISABLE','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH']}\n   assert selected_env['GGML_CUDA_DISABLE_GRAPHS'] in [None,'0'],'Do not handicap the competitor by disabling graphs'\n   (directory/'actual-environment.json').write_text(json.dumps(selected_env,indent=2)+chr(10))\n   maps=pathlib.Path(f'/proc/{server.pid}/maps').read_text().splitlines();paths={pathlib.Path(line.split()[-1]) for line in maps if '.so' in line and any(k in line.split()[-1] for k in ['libllama','libggml','libnccl']) and line.split()[-1].startswith('/')}\n   (directory/'loaded-library-pins.json').write_text(json.dumps({str(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(paths)},indent=2)+chr(10))\n  cmd=[py,'-u','tools/benchmark_completion_api.py'")
source=source.replace(" finally:\n  stop(server)", " finally:\n  item['server_exit_code_before_cleanup']=server.poll() if server else None\n  stop(server)")
source+='\nEXPECTED='+repr(json.loads((b/'source-manifest.json').read_text()))+'\nHISTORICAL='+repr(json.loads((b/'historical-external-audit.json').read_text()))+r'''
try:
 prior=root/'artifacts/venturi-native-head-controls-v1';dependent=root/'artifacts/venturi-native-head-aligned-v2'
 assert (prior/'exit-code.txt').read_text().strip()=='1' and (dependent/'exit-code.txt').read_text().strip()=='1'
 failed_log=(prior/'fresh-llama-tensor-off/server.log').read_text();assert 'error while loading shared libraries:' in failed_log and 'cannot open shared object file' in failed_log,'Recovery must match observed dependency failure'
 previous=json.loads((prior/'campaign-report.json').read_text());assert previous['path_promotions']['ordinary256'] and len(previous['experiments'])==7
 assert all(x['completed'] for x in previous['experiments'][:6]) and not previous['experiments'][-1]['completed']
 q=prior/'qualification256/report.json';assert json.loads(q.read_text())['completed']
 library=pathlib.Path((prior/'library.txt').read_text().strip());assert digest(library)=='33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8';(out/'library.txt').write_text(str(library)+chr(10))
 actual={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==EXPECTED;(out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf';bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin')
 def identity(p):
  s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 before={str(p):identity(p) for p in [target,draft,gguf,bindir/'llama-server']};pins=json.loads((prior/'fresh-external-pins.json').read_text());assert before==pins['identity']==json.loads((prior/'payload-before.json').read_text())
 assert digest(bindir/'llama-server')==pins['binary_sha256'];(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10));(out/'fresh-external-pins.json').write_text(json.dumps(dict(pins,reused_full_gguf_scan_from=str(prior/'fresh-external-pins.json'),prior_report_sha256=digest(prior/'fresh-external-pins.json')),indent=2)+chr(10))
 ldd_env=env.copy();ldd_env['LD_LIBRARY_PATH']=str(bindir)+':/usr/local/cuda/lib64:'+str(root/'repos/codex-build/.venvs/b2-v100/lib/python3.10/site-packages/nvidia/nccl/lib')
 ldd=subprocess.run(['ldd',str(bindir/'llama-server')],capture_output=True,text=True,env=ldd_env,check=True);(out/'resolved-ldd.txt').write_text(ldd.stdout+ldd.stderr);assert 'not found' not in ldd.stdout
 env['NCCL_P2P_DISABLE']='1';selected_rows=32;selected_profile=PROFILE;selected_draft='q8-residual'
 report.update(scope='Final native/external matched controls and bounded parallel MTP-depth search; unchanged native implementation and passed ordinary256 gates reused.',library_sha256=digest(library),source_sha256=actual,path_promotions={'ordinary256':True},qualified256_sha256=digest(q),reused_ordinary256_report=dict(path=str(prior/'campaign-report.json'),sha256=digest(prior/'campaign-report.json')),fresh_external_campaign_run=True)
 for label,mode,depth,counts in [('fresh-llama-tensor-off','tensor',0,'1,2,4,8'),('fresh-llama-layer-mtp1','layer',1,'1,2,4,8'),('fresh-llama-tensor-mtp1-c1','tensor',1,'1'),('fresh-llama-tensor-mtp2-c1','tensor',2,'1')]:
  experiment(label,'llama',mode,depth,counts,3)
 # Address the remaining competitor-search gap: deeper layer-split MTP at
 # parallel loads and tensor MTP at C2/C4, never retry the known C8 crash.
 layer_screens=[]
 for depth in [2,3,4]:
  value=experiment(f'screen-llama-layer-mtp{depth}','llama','layer',depth,'1,8',1)
  layer_screens.append(value)
 best_layer=max(layer_screens,key=lambda e:next(r['aggregate_complete_request_output_tps'] for r in e['summary'] if r['request_count']==8))
 report['layer_parallel_depth_screen_winner']=best_layer['mtp_depth'];save()
 experiment(f'repeat-llama-layer-mtp{best_layer["mtp_depth"]}','llama','layer',best_layer['mtp_depth'],'1,2,4,8',3)
 tensor_screens=[]
 for depth in [1,2,3,4]:
  try:value=experiment(f'screen-llama-tensor-mtp{depth}-c2c4','llama','tensor',depth,'2,4',1)
  except BaseException:
   failure=report['experiments'][-1]
   if failure['server_exit_code_before_cleanup']==-11:
    report['tensor_parallel_mtp_multirequest_sigsegv']=dict(depth=depth,failed_label=failure['label'],not_counted_as_zero_tps=True,skip_same_failure_family_deeper_depths=True);save();break
   raise
  tensor_screens.append(value)
 if tensor_screens:
  winners={max(tensor_screens,key=lambda e:next(r['aggregate_complete_request_output_tps'] for r in e['summary'] if r['request_count']==count))['mtp_depth'] for count in [2,4]}
  for depth in sorted(winners):experiment(f'repeat-llama-tensor-mtp{depth}-c2c4','llama','tensor',depth,'2,4',3)
 experiment('confirm-native-mtp2-c1','venturi','scalar',2,'1',3,'prefix')
 experiment('confirm-native-mtp1-c8','venturi','mtp-batch',1,'8',3)
 after={str(p):identity(p) for p in [target,draft,gguf,bindir/'llama-server']};assert after==before;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 report.update(completed=True,stronger_historical_controls_retained=True,full_goal_complete=False,requires_completion_audit=True,source_dependency_failure_recovered_without_repeating_native_qualification_or_matrices=True);save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 try:run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 except BaseException:(out/'cleanup-observation-failure.txt').write_text(traceback.format_exc())
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
compile(source,'worker','exec');Path('/tmp/venturi-native-head-controls-v4-source.py').write_text(source)
