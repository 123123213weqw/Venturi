from pathlib import Path
import json
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);out=Path('benchmarks/v100-native-head-gdn')
old=Path('/tmp/venturi-mtp-draft-residual-campaign-v3-worker.py').read_text();functions=old[old.index('def save():'):old.index('\ntry:\n deadline=')]
functions=functions.replace("candidate_build=(selected_draft=='q8-residual')", "candidate_build=(selected_profile==PROFILE)")
functions=functions.replace(" directory=out/label;directory.mkdir();", " assert not any(':18953' in row for row in subprocess.run(['ss','-ltnp'],capture_output=True,text=True,check=True).stdout.splitlines()),'Owned port already occupied; do not stop an unrelated process'\n directory=out/label;directory.mkdir();")
functions=functions.replace("  assert item['health']['mtp_draft_projection_profile']==selected_draft", "  assert item['health']['mtp_draft_projection_profile']==selected_draft\n  assert item['health']['projection_profile']==selected_profile")
head=r'''import pathlib,subprocess,json,os,time,traceback,signal,urllib.request,hashlib,datetime,statistics
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-native-head-gdn-campaign-v5';src=root/'repos/codex-build/Venturi/native-head-gdn-src-v3';prior=root/'artifacts/venturi-native-head-gdn-mtp-v4'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';port=18953
PROFILE='q8-residual-column-mlp-head-gdn';CONTROL='q8-residual-column-mlp'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
report=dict(completed=False,scope='Qualified native semantic-head GDN target versus accepted row-GDN/column-MLP target; identical library, Q8 payload/scales, residual draft, FP16 KV, IDs/work/context. New GDN output FP32 association separately qualified. Complete HTTP versus native decode scopes remain separate.',experiments=[],physical_gpus=[1,0],utc_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds()),achieved_hardware_counters=None,path_promotions={},screen_ratios={})
'''
body=r'''
try:
 deadline=time.monotonic()+5700
 while not (prior/'exit-code.txt').exists():
  pid=json.loads((prior/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(prior/'worker.py').encode() in q.read_bytes(),'Prerequisite missing without terminal marker';assert time.monotonic()<deadline,'Verified live prerequisite pending; do not restart';time.sleep(5)
 assert (prior/'exit-code.txt').read_text().strip()=='0','MTP qualification failed'
 for sub in ['mtp2-qualification','mtp-batch-qualification']:assert json.loads((prior/sub/'report.json').read_text())['completed']
 actual={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==json.loads((prior/'executed-source.json').read_text())
 library=pathlib.Path((prior/'library.txt').read_text().strip());candidate_library=library;build=json.loads(library.with_name('build.json').read_text());assert digest(library)==build['library_sha256']
 for rel,h in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==h
 report.update(model_qualified=True,source_sha256=actual,library_sha256=digest(library),target_report_pin=json.loads((prior/'reused-target-qualification.json').read_text()),fresh_external_campaign_run=False,external_goal_complete=False)
 def identity(p):
  s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';before={str(p):identity(p) for p in [target,draft]};assert before==json.loads((prior/'payload-after.json').read_text());(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 selected_rows=32;selected_draft='q8-residual';paths=[('mtp2','scalar',2,'prefix'),('mtp-batch','mtp-batch',1,'replay')]
 for name,mode,depth,recovery in paths:
  screens=[]
  for trial in range(2):
   values={}
   for selected_profile in ([CONTROL,PROFILE] if trial==0 else [PROFILE,CONTROL]):
    arm='candidate' if selected_profile==PROFILE else 'control';values[arm]=experiment(f'screen-{name}-{trial+1}-{arm}','venturi',mode,depth,'1,8',1,recovery)
   by_arm={arm:{r['request_count']:r for r in value['summary']} for arm,value in values.items()};metric='aggregate_complete_request_output_tps';screens.append({str(count):by_arm['candidate'][count][metric]/by_arm['control'][count][metric] for count in (1,8)})
  median={str(count):statistics.median(s[str(count)] for s in screens) for count in (1,8)}
  report['screen_ratios'][name]=dict(paired_trials=screens,median=median)
  report['path_promotions'][name]=all(v>=.98 for v in median.values()) and any(median[str(count)]>=1.01 and all(s[str(count)]>1 for s in screens) for count in (1,8));save()
 for name,mode,depth,recovery in paths:
  if not report['path_promotions'][name]:continue
  for selected_profile in [CONTROL,PROFILE]:
   arm='candidate' if selected_profile==PROFILE else 'control';experiment(f'matrix-{name}-{arm}','venturi',mode,depth,'1,2,4,8',3,recovery)
 report['stronger_historical_controls_retained']=True;report['new_profile_ordinary256_not_tested']=True
 report['promotion_scope']='Two alternating C1/C8 screens; >=1% median in one cell with both paired trials improving, and no other cell worse than0.98. Only promoted paths receive three-repeat1/2/4/8 matrices. No model TPS extrapolation, no full goal completion claim.'
 after={str(p):identity(p) for p in [target,draft]};assert after==before;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10));report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 try:run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 except BaseException:(out/'cleanup-observation-failure.txt').write_text(traceback.format_exc())
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
source=head+functions+body
(out/'campaign-v5-launcher.py').write_text(Path(__file__).read_text())
n['dispatch']('venturi-native-head-gdn-campaign-v5',source)
