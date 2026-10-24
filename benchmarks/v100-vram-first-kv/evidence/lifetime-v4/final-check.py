import hashlib,json,subprocess,time,urllib.request
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
P=Path('/home/data/wangyue/artifacts/venturi-kv-tier-lifetime-v4')
spec=spec_from_file_location('restorer','/home/data/wangyue/repos/codex-build/Venturi/kv-tier-control-final-v3/restore_final.py');m=module_from_spec(spec);spec.loader.exec_module(m)
a=m.inspect();before=json.loads(Path('/home/data/wangyue/artifacts/venturi-kv-tier-window-final-v3/production-before.json').read_text());assert a['Id']==before['container_id'] and m.fingerprint(a)==before['canonical_config_fingerprint'];assert a['State']['Running'] and a['State']['Health']['Status']=='healthy'
def api(n):return json.load(urllib.request.urlopen('http://127.0.0.1:8000/'+n,timeout=8))
health,models=api('health'),api('v1/models');assert health['status']=='ok'
units=['venturi-kv-tier-window-v1','venturi-kv-tier-window-final-v3','venturi-kv-tier-build-v1','venturi-kv-tier-build-v2','venturi-kv-tier-build-v3','venturi-kv-tier-lifetime-v4','venturi-kv-tier-restore-v1.timer','venturi-kv-tier-restore-final-v3.timer']
states={u:subprocess.check_output(['systemctl','--user','show',u,'-p','ActiveState','--value'],text=True,timeout=5).strip() for u in units};assert all(x in ['inactive','failed'] for x in states.values()),states
policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s];assert policy==before['counter_policy']
model=models['data'][0]['id'];body=json.dumps(dict(model=model,messages=[dict(role='user',content='Return OK.')],max_tokens=4,temperature=0)).encode();q=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
t=time.monotonic()
with urllib.request.urlopen(q,timeout=60) as response:status=response.status;result=json.load(response)
assert status==200 and result.get('choices')
message=result['choices'][0].get('message',{});assert message.get('content') or message.get('reasoning_content')
r=dict(healthy=True,config_unchanged=True,container_id=a['Id'],canonical_config_fingerprint=m.fingerprint(a),health_api=health,models_api=models,unit_states=states,counter_policy=policy,production_completion=dict(http_status=status,wall_seconds=time.monotonic()-t,finish_reason=result['choices'][0].get('finish_reason'),response_nonempty=True,usage=result.get('usage')),own_timeout_dropin_removed=not Path('/run/user/1000/systemd/user/venturi-kv-tier-window-v1.service.d/venturi-timeout.conf').exists(),unix=time.time())
(P/'final-production-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
