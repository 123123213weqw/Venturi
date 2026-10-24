"""Read-only profiler aggregation and final exact-container/API verification."""
import hashlib,importlib.util,json,pathlib,sqlite3,subprocess,sys,time,urllib.request
R=pathlib.Path('/home/data/wangyue/artifacts/venturi-rms-timeline-repair-v1')
H=pathlib.Path('/home/data/wangyue/repos/codex-build/Venturi/rms-native-src-v7/tools/summarize_native_runtime_trace.py')
spec=importlib.util.spec_from_file_location('trace_helper',H);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
result=dict(scope='Profiler-only four-token native batch-8 post-first region. No profiler durations used as TPS scores. CUDA event completion tracing can perturb scheduling; gaps are not device-wide idle or proof of CPU causality.',traces={})
for arm in ['off','on']:
 p=R/'transfer-windows'/('timeline-'+arm+'.sqlite');db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 try:
  strings=dict(db.execute('select id,value from StringIds'));tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
  ks,cs,ms=[list(db.execute('select * from '+t)) if t in tables else [] for t in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']]
  all=h.activity_summary(ks,cs,ms,strings);all['kernel_by_name']=all['kernel_by_name'][:6]
  devices=[]
  for gpu in [0,1]:
   s=h.activity_summary(*[[r for r in group if r['deviceId']==gpu] for group in [ks,cs,ms]],strings);s['kernel_by_name']=s['kernel_by_name'][:6];devices.append(dict(device_id=gpu,**s))
  result['traces'][arm]=dict(filtered_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),aggregate=all,by_device=devices)
 finally:db.close()
sys.path.insert(0,'/home/data/wangyue/repos/codex-build/Venturi/rms-native-window-control-v4')
from restore_service_v1 import inspect,fingerprint
before=json.loads((R/'production-before.json').read_text());a=inspect();assert a['Id']==before['container_id'] and fingerprint(a)==before['canonical_config_fingerprint'] and a['State']['Health']['Status']=='healthy'
health=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5));assert health.get('status')=='ok'
models=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5));model=models['data'][0]['id']
cleanup=subprocess.run(['systemctl','--user','unmask','--runtime','venturi-rms-model-window-v3.service'],capture_output=True,text=True,timeout=10);assert cleanup.returncode==0
body=dict(model=model,messages=[dict(role='user',content='Reply with OK only.')],max_tokens=8,temperature=0,stream=False,chat_template_kwargs=dict(enable_thinking=False))
request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
response=json.load(urllib.request.urlopen(request,timeout=40));assert response.get('choices')
result['production']=dict(healthy=True,container_id=a['Id'],canonical_config_fingerprint=fingerprint(a),config_unchanged=True,model=model,health=health,smoke_response=response,own_abort_mask_removed=True,unix=time.time(),counter_policy=[s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s]);assert result['production']['counter_policy']==before['counter_policy']
(R/'final-read-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
