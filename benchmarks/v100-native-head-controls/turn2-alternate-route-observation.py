from pathlib import Path
import subprocess,json,shlex
b=Path('benchmarks/v100-native-head-controls');opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
code="""import pathlib,json,socket
base=pathlib.Path('/home/data/wangyue/artifacts');o=base/'venturi-native-head-controls-v1';p=base/'venturi-native-head-controls-v4';print(json.dumps(dict(host=socket.gethostname(),server_log=(o/'fresh-llama-tensor-off/server.log').read_text(),v4_exists=p.exists(),v4_process=(p/'process.json').read_text() if (p/'process.json').exists() else None)))
"""
proxy=shlex.join(['ssh']+opts+['-W','%h:%p','School_4090D'])
routes=[('rescue',['ssh']+opts+['WZU_Server_Rescue','python3 -']),('campus',['ssh']+opts+['-o','ProxyCommand='+proxy,'V100_LAN_J','python3 -'])]
for label,ssh in routes:
 try:r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=65)
 except subprocess.TimeoutExpired as e:(b/f'turn2-{label}-route.log').write_text(str(e));print(label,'bounded timeout',flush=True);continue
 (b/f'turn2-{label}-route.log').write_text(r.stdout+r.stderr);print(label,r.returncode,r.stdout,r.stderr,flush=True)
 if not r.returncode:(b/'turn2-healthy-route.json').write_text(json.dumps(dict(ssh=ssh,observation=json.loads(r.stdout)),indent=2)+'\n');break
