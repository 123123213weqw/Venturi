from pathlib import Path
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
s=Path('benchmarks/v100-native-column-mlp/v100/campaign-v6/worker.py').read_text();functions=s[s.index('def save():'):s.index('candidate_library=library')]
functions=functions.replace("projection_profile=selected_profile if engine=='venturi' else None,","projection_profile=selected_profile if engine=='venturi' else None,mtp_draft_projection_profile=selected_draft if engine=='venturi' and depth else None,")
functions=functions.replace("candidate_build=(library==candidate_library)","candidate_build=(selected_draft=='q8-residual')")
functions=functions.replace("   if depth:cmd+=['--mtp-artifact'","   if depth:cmd+=['--mtp-draft-projection-profile',selected_draft,'--mtp-artifact'")
functions=functions.replace("(directory/'loaded-maps.txt').write_text", "assert item['health']['mtp_draft_projection_profile']==selected_draft\n  (directory/'loaded-maps.txt').write_text")
head=r'''import pathlib,subprocess,json,os,time,traceback,signal,urllib.request,hashlib,datetime
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-mtp-draft-residual-campaign-v3';src=root/'repos/codex-build/Venturi/mtp-draft-residual-src-v1';prior=root/'artifacts/venturi-mtp-draft-residual-model-v2'
py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json';port=18953
bindir=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin');gguf=root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,0',PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps')+':'+str(src/'src')+':'+str(src/'tests'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',NCCL_P2P_DISABLE='1',LD_LIBRARY_PATH=str(bindir)+':/usr/local/cuda/lib64:'+str(root/'repos/codex-build/.venvs/b2-v100/lib/python3.10/site-packages/nvidia/nccl/lib'))
report=dict(completed=False,scope='Explicit residual-A8 draft vs original FP16 draft; same native library/SDK, qualified column-MLP target, fixed Q8 weights/FP16 KV/work/policies. Complete HTTP cohort includes reset/admission/replies/client archival; native decode excludes those. Changed draft arithmetic separately qualified; not old-profile byte parity or BF16 corpus quality.',experiments=[],physical_gpus=[1,0],utc_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds()),achieved_hardware_counters=None,path_promotions={},screen_ratios={})
'''
body=r'''
try:
 deadline=time.monotonic()+3600
 while not (prior/'exit-code.txt').exists():
  pid=json.loads((prior/'process.json').read_text())['pid'];cmd=pathlib.Path(f'/proc/{pid}/cmdline');assert cmd.exists() and str(prior/'worker.py').encode() in cmd.read_bytes(),'Prerequisite missing without terminal file'
  assert time.monotonic()<deadline,'Bounded wait elapsed with prerequisite still live';time.sleep(3)
 assert (prior/'exit-code.txt').read_text().strip()=='0','Model qualification failed'
 actual={str(p.relative_to(src)):digest(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==json.loads((prior/'executed-source.json').read_text())
 library=pathlib.Path((prior/'library.txt').read_text().strip());candidate_library=library
 for profile in ['fp16','q8-residual']:
  q=json.loads((prior/(profile+'-qualification/report.json')).read_text());assert q['completed'] and all(v['passed'] for v in q['block_reference']+q['recurrence_reference'])
  assert all(v['output_ids_equal'] and v['committed_target_state_bytes_equal'] and v['committed_draft_state_equal_mtp1'] for v in q['cases'])
 q=json.loads((prior/'batch-qualification/report.json').read_text());assert q['completed'] and q['selective_reset_untouched_state_exact']
 build=json.loads(library.with_name('build.json').read_text());assert digest(library)==build['library_sha256']
 for rel,h in build['source_sha256'].items():assert digest(src/'src/venturi/ops/csrc'/rel)==h
 report.update(model_qualified=True,source_sha256=actual,candidate_library_sha256=digest(library),projection_profiles=dict(target='q8-residual-column-mlp',control_draft='fp16',candidate_draft='q8-residual'))
 selected_profile='q8-residual-column-mlp';selected_rows=32
 paths=[('mtp2','scalar',2,'prefix'),('mtp-batch','mtp-batch',1,'replay')]
 for name,mode,depth,recovery in paths:
  selected_draft='fp16';control=experiment('screen-control-'+name,'venturi',mode,depth,'1,8',1,recovery)
  selected_draft='q8-residual';candidate=experiment('screen-candidate-'+name,'venturi',mode,depth,'1,8',1,recovery)
  a={r['request_count']:r for r in control['summary']};b={r['request_count']:r for r in candidate['summary']};metric='aggregate_complete_request_output_tps'
  ratios={str(n):b[n][metric]/a[n][metric] for n in b};report['screen_ratios'][name]=ratios;report['path_promotions'][name]=ratios['1']>1.01 or ratios['8']>1.02;save()
 for name,mode,depth,recovery in paths:
  if not report['path_promotions'][name]:continue
  selected_draft='fp16';experiment('repeat-control-'+name,'venturi',mode,depth,'1,2,4,8',3,recovery)
  selected_draft='q8-residual';experiment('repeat-candidate-'+name,'venturi',mode,depth,'1,2,4,8',3,recovery)
 report['reused_controls']=[]
 for p in [root/'artifacts/venturi-native-column-mlp-campaign-v6/campaign-report.json',root/'artifacts/venturi-wmma-campaign-v5/campaign-report.json',root/'artifacts/venturi-strided-external-depth-v9/campaign-report.json']:
  assert json.loads(p.read_text())['completed'];report['reused_controls'].append(dict(path=str(p),sha256=digest(p)))
 report['ordinary256_unaffected_and_prior_stronger_control_retained']=True
 report['promotion_rule']='C1 HTTP >1% or C8 >2% admits three-repeat1/2/4/8; subsequent adoption requires no repeated cell worse than0.98 and two improving qualifying trials. Full external goal is unchanged.'
 report['fresh_external_campaign_run']=False;report['external_goal_complete']=False;report['completed']=True;save();rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 try:run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],out/'gpu-after.txt')
 except BaseException:(out/'cleanup-observation-failure.txt').write_text(traceback.format_exc())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
ns['dispatch']('venturi-mtp-draft-residual-campaign-v3',head+functions+body)
