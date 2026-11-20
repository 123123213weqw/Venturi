from pathlib import Path
import subprocess,json,ast
b=Path('benchmarks/v100-native-head-controls');ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=/Users/wangyue/.ssh/cm-14974355c3f315e78f58843cae0595e186514110','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=6','WZU_Server','python3 -']
check=subprocess.run(['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-O','check','WZU_Server'],capture_output=True,text=True,timeout=10);print('Control check',check.returncode,check.stdout,check.stderr,flush=True)
(b/'turn2-master-check.log').write_text(check.stdout+check.stderr)
if not check.returncode:
 code="""import pathlib,json
base=pathlib.Path('/home/data/wangyue/artifacts');o=base/'venturi-native-head-controls-v1';p=base/'venturi-native-head-controls-v4';print(json.dumps(dict(server_log=(o/'fresh-llama-tensor-off/server.log').read_text(),v4_exists=p.exists(),v4_process=(p/'process.json').read_text() if (p/'process.json').exists() else None)))
"""
 try:r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=65)
 except subprocess.TimeoutExpired as e:(b/'turn2-master-recovery.log').write_text(str(e));print('Bounded master read timeout; no relaunch',flush=True);raise SystemExit(0)
 (b/'turn2-master-recovery.log').write_text(r.stdout+r.stderr);print(r.returncode,r.stdout,r.stderr,flush=True)
 if not r.returncode:
  d=json.loads(r.stdout)
  if not d['v4_exists']:
   tree=ast.parse(Path('/tmp/venturi-native-head-controls-v4-dispatch.py').read_text());source=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(getattr(t,'id','')=='code' for t in node.targets))
   q=subprocess.run(ssh,input=source,text=True,capture_output=True,timeout=80);(b/'v4-master-dispatch.log').write_text(q.stdout+q.stderr);print('Verified absent; exact dispatch',q.returncode,q.stdout,q.stderr,flush=True)
  else:print('Existing v4; do not dispatch again',flush=True)
