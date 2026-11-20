import subprocess,json
from pathlib import Path
b=Path('benchmarks/v100-native-head-controls');n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
# Reuse the pre-existing authorized same-host verified control channel.
ssh=n['SSH'][:];i=ssh.index('ControlPath=none');ssh[i]='ControlPath=/Users/wangyue/.ssh/cm-14974355c3f315e78f58843cae0595e186514110'
code="""import json,pathlib,socket
p=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-head-controls-v1');r=dict(host=socket.gethostname(),artifact_exists=p.exists())
if p.exists():
 for name in ['process.json','exit-code.txt','failure.txt']:
  if (p/name).exists():r[name]=(p/name).read_text()
print(json.dumps(r))
"""
try:
 r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=40);(b/'control-channel-observation.log').write_text(r.stdout+r.stderr);print('MUX',r.returncode,r.stdout,r.stderr,flush=True)
except subprocess.TimeoutExpired as e:(b/'control-channel-observation.log').write_text(str(e));print('Existing master not responsive within40s',flush=True)
try:
 r=subprocess.run(['tailscale','status','--json'],capture_output=True,text=True,timeout=15);d=json.loads(r.stdout);selected={k:{x:v.get(x) for x in ['HostName','DNSName','TailscaleIPs','Online','CurAddr','Relay','LastSeen','Active']} for k,v in d.get('Peer',{}).items() if '100.70.188.115' in v.get('TailscaleIPs',[])};o=dict(backend=d.get('BackendState'),self_online=d.get('Self',{}).get('Online'),target=selected);(b/'tailscale-target-status.json').write_text(json.dumps(o,indent=2)+'\n');print(o,flush=True)
except Exception as e:print('Tailscale status unavailable',type(e).__name__,flush=True)
