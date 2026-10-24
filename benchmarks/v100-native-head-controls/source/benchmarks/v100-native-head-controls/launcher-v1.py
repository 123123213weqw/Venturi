from pathlib import Path
import json
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
b=Path('benchmarks/v100-native-head-controls');old=Path('benchmarks/v100-native-head-gdn/v100/campaign-v5/worker.py').read_text()
source=old[:old.index('\ntry:\n deadline=')].replace('venturi-native-head-gdn-campaign-v5','venturi-native-head-controls-v1')
source=source.replace("assert item['health']['mtp_draft_projection_profile']==selected_draft\n  assert item['health']['projection_profile']==selected_profile", "if engine=='venturi':\n   if depth:assert item['health']['mtp_draft_projection_profile']==selected_draft\n   assert item['health']['projection_profile']==selected_profile")
source=source.replace("run(cmd,directory/'benchmark.log',timeout=600)","run(cmd if engine=='venturi' else cmd[:-1],directory/'benchmark.log',timeout=600)")
source=source.replace("return hashlib.sha256(path.read_bytes()).hexdigest()", "\n h=hashlib.sha256()\n with path.open('rb') as f:\n  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)\n return h.hexdigest()")
source+="\nEXPECTED="+repr(json.loads((b/'source-manifest.json').read_text()))+'\nQUALIFIER='+repr((b/'qualify_head_profile.py').read_text())+r'''
try:
 env['NCCL_P2P_DISABLE']='1'
 expected_library='33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8'
 previous=root/'artifacts/venturi-native-head-gdn-model-v3';qualified32=previous/'qualification/report.json'
 assert digest(qualified32)=='6e4c1623bc10488bbcb92f92d26d9f0346c3f727db65673acb9808e3970dba50'
 library=pathlib.Path((previous/'library.txt').read_text().strip());assert digest(library)==expected_library
 actual={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==EXPECTED
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10));(out/'library.txt').write_text(str(library)+chr(10))
 build=json.loads(library.with_name('build.json').read_text());assert len(build['source_sha256'])==11
 for rel,h in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==h
 def identity(p):
  s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'
 gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf'
 bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin')
 before={str(p):identity(p) for p in [target,draft,gguf,bindir/'llama-server']};(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 assert digest(bindir/'llama-server')=='e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2'
 assert digest(gguf)=='eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513'
 (out/'fresh-external-pins.json').write_text(json.dumps(dict(binary_sha256=digest(bindir/'llama-server'),gguf_sha256=digest(gguf),identity=before,source_commit_unknown=True,graphs_not_disabled=True,NCCL_P2P_DISABLE='1'),indent=2)+chr(10))
 env['PYTHONPATH']+=':'+str(src/'tools')
 teacher=root/'artifacts/venturi-native-column-mlp-model-v2/qualification/report.json';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json'
 qscript=out/'qualify_head_profile.py';qscript.write_text(QUALIFIER)
 run([py,str(qscript),str(target),'--library',str(library),'--cases',str(cases),'--teacher-report',str(teacher),'--teacher-report-sha256','904bace8b87a2047191e4fdeab66a420dfacb100d5d9686beb1ef52ac0d655f7','--projection-rows','256','--output-directory',str(out/'qualification256')],out/'qualification256.log',timeout=2700)
 q=json.loads((out/'qualification256/report.json').read_text());assert q['completed'] and q['profiles'][PROFILE]['candidate']['passed']
 report.update(library_sha256=digest(library),qualified32_sha256=digest(qualified32),qualified256_sha256=digest(out/'qualification256/report.json'),fresh_external_campaign_run=True,source_sha256=actual)
 selected_rows=256;selected_draft='q8-residual';screens=[]
 for trial in range(2):
  values={}
  for selected_profile in ([CONTROL,PROFILE] if trial==0 else [PROFILE,CONTROL]):
   arm='candidate' if selected_profile==PROFILE else 'control';values[arm]=experiment(f'screen-ordinary256-{trial+1}-{arm}','venturi','batch',0,'1,8',1)
  by_arm={arm:{r['request_count']:r for r in v['summary']} for arm,v in values.items()};screens.append({str(c):by_arm['candidate'][c]['aggregate_complete_request_output_tps']/by_arm['control'][c]['aggregate_complete_request_output_tps'] for c in (1,8)})
 median={str(c):statistics.median(s[str(c)] for s in screens) for c in (1,8)}
 promoted=all(v>=.98 for v in median.values()) and any(median[str(c)]>=1.01 and all(s[str(c)]>1 for s in screens) for c in (1,8))
 report['screen_ratios']['ordinary256']=dict(paired_trials=screens,median=median);report['path_promotions']['ordinary256']=promoted;save()
 if promoted:
  for selected_profile in [CONTROL,PROFILE]:
   arm='candidate' if selected_profile==PROFILE else 'control';experiment('matrix-ordinary256-'+arm,'venturi','batch',0,'1,2,4,8',3)
 # Do not re-run the unchanged known tensor-MTP1 C8 crash. Valid alternatives
 # retain CUDA graph/default optimizations and all earlier stronger baselines.
 for label,mode,depth,counts in [
  ('fresh-llama-tensor-off','tensor',0,'1,2,4,8'),
  ('fresh-llama-layer-mtp1','layer',1,'1,2,4,8'),
  ('fresh-llama-tensor-mtp1-c1','tensor',1,'1'),
  ('fresh-llama-tensor-mtp2-c1','tensor',2,'1')]:
  experiment(label,'llama',mode,depth,counts,3)
 selected_rows=32;selected_profile=PROFILE
 experiment('confirm-native-mtp2-c1','venturi','scalar',2,'1',3,'prefix')
 experiment('confirm-native-mtp1-c8','venturi','mtp-batch',1,'8',3)
 after={str(p):identity(p) for p in [target,draft,gguf,bindir/'llama-server']};assert after==before
 (out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 report.update(completed=True,stronger_historical_controls_retained=True,full_goal_complete=False,requires_completion_audit=True);save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 try:run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 except BaseException:(out/'cleanup-observation-failure.txt').write_text(traceback.format_exc())
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
(b/'launcher-v1.py').write_text(Path(__file__).read_text());n['dispatch']('venturi-native-head-controls-v1',source)
