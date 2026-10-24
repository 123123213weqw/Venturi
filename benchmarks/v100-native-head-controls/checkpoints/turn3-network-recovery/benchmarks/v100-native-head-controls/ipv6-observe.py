from pathlib import Path
import subprocess,json,time
b=Path('benchmarks/v100-native-head-controls');status=subprocess.run(['tailscale','status','--json'],capture_output=True,text=True,timeout=20);d=json.loads(status.stdout);peer=next(v for v in d['Peer'].values() if '100.70.188.115' in v.get('TailscaleIPs',[]));ipv6=next(x for x in peer['TailscaleIPs'] if ':' in x)
(b/'turn3-target-addresses.json').write_text(json.dumps({k:peer.get(k) for k in ['HostName','DNSName','TailscaleIPs','Online','Relay']},indent=2)+'\n')
# The authenticated peer address changes, not the expected verified SSH identity.
known=subprocess.run(['ssh-keygen','-F','[100.70.188.115]:9022'],capture_output=True,text=True,timeout=10);assert known.returncode==0,'Do not bypass missing host verification'
ssh=['ssh','-6','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','HostName='+ipv6,'-o','HostKeyAlias=[100.70.188.115]:9022','WZU_Server','python3 -']
code="""import pathlib,json,socket
base=pathlib.Path('/home/data/wangyue/artifacts');r={'host':socket.gethostname()}
for name in ['venturi-native-head-controls-v4','venturi-native-head-aligned-v5']:
 p=base/name;d={'exists':p.exists()}
 for f in ['process.json','exit-code.txt','failure.txt']:
  if (p/f).exists():d[f]=(p/f).read_text()[-2000:]
 if (p/'process.json').exists():
  q=pathlib.Path('/proc')/str(json.loads((p/'process.json').read_text())['pid'])/'cmdline';d['live_cmdline']=q.read_bytes().decode(errors='replace') if q.exists() else None
 r[name]=d
print(json.dumps(r))
"""
try:r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=45)
except subprocess.TimeoutExpired as e:(b/'turn3-ipv6-observation.log').write_text(str(e));print('IPv6 observation timeout; nothing dispatched',flush=True);raise SystemExit(0)
(b/'turn3-ipv6-observation.log').write_text(r.stdout+r.stderr);print(r.returncode,r.stdout,r.stderr,flush=True)
if not r.returncode:
 data=json.loads(r.stdout);assert data['host']=='wzu-SYS-4029GP-TRT';(b/'turn3-ipv6-authoritative-state.json').write_text(json.dumps(data,indent=2)+'\n');(b/'turn3-ipv6-healthy-route.json').write_text(json.dumps(dict(ssh=ssh,observed_unix=time.time(),expected_host_key_identity='[100.70.188.115]:9022'),indent=2)+'\n')
