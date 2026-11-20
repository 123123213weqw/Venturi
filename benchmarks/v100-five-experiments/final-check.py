import hashlib,json,subprocess,time,urllib.request
from pathlib import Path
R=Path('/home/data/wangyue/artifacts/venturi-five-window-v4');N=Path('/home/data/wangyue/artifacts/venturi-five-node-repair-v1')
def api(name):return json.load(urllib.request.urlopen('http://127.0.0.1:8000/'+name,timeout=6))
a=json.loads(subprocess.check_output(['docker','inspect','qwen38-27b'],timeout=12))[0]
f=hashlib.sha256(json.dumps(dict(args=a['Args'],env=a['Config']['Env'],mounts=sorted(a['Mounts'],key=lambda v:(v.get('Destination',''),v.get('Source',''),v.get('Type','')))),sort_keys=True).encode()).hexdigest()
before=json.loads((R/'production-before.json').read_text());assert a['Id']==before['container_id'] and f==before['canonical_config_fingerprint']
assert a['State']['Running'] and a['State']['Health']['Status']=='healthy' and api('health')['status']=='ok'
models=api('v1/models');model=models['data'][0]['id']
body=dict(model=model,messages=[dict(role='user',content='Reply with OK.')],max_tokens=8,temperature=0,stream=False,chat_template_kwargs=dict(enable_thinking=False))
r=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=35) as response:status=response.status;value=json.load(response)
assert status==200 and value.get('choices') and value.get('usage',{}).get('completion_tokens',0)>0
units=subprocess.check_output(['systemctl','--user','show','venturi-five-build-v2','venturi-five-window-v4','venturi-five-node-repair-v1','venturi-five-restore-v4.timer','venturi-five-node-restore-v1.timer','-p','Id','-p','ActiveState','-p','SubState','-p','ExecMainStatus'],text=True,timeout=10)
old=subprocess.check_output(['systemctl','--user','show','venturi-rms-model-window-v3','-p','LoadState','-p','ActiveState'],text=True,timeout=10)
cleanup=None
if 'LoadState=masked' in old:
 c=subprocess.run(['systemctl','--user','unmask','--runtime','venturi-rms-model-window-v3.service'],capture_output=True,text=True,timeout=10);cleanup=dict(rc=c.returncode,output=c.stdout+c.stderr,unit_started=False)
policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s];assert policy==before['counter_policy']
out=dict(healthy=True,container_id=a['Id'],config_fingerprint=f,config_unchanged=True,started_at=a['State']['StartedAt'],models_api=models,http_smoke=dict(status=status,model=value.get('model'),completion_tokens=value['usage']['completion_tokens'],finish_reason=value['choices'][0].get('finish_reason'),content=value['choices'][0]['message'].get('content')),counter_policy=policy,units=units,old_own_mask=old,old_own_mask_cleanup=cleanup,unix=time.time())
p=N/'final-production-check.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
