from pathlib import Path
import subprocess,json,time
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);ssh=n['SSH'][:];ssh[ssh.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6'
b=Path('benchmarks/v100-native-head-controls');code="""import pathlib,json
base=pathlib.Path('/home/data/wangyue/artifacts');o=base/'venturi-native-head-controls-v1';p=base/'venturi-native-head-controls-v4';print(json.dumps(dict(server_log=(o/'fresh-llama-tensor-off/server.log').read_text(),v1_rc=(o/'exit-code.txt').read_text(),v2_rc=(base/'venturi-native-head-aligned-v2/exit-code.txt').read_text(),v4_exists=p.exists(),v4_process=(p/'process.json').read_text() if (p/'process.json').exists() else None)))
"""
for attempt in range(3):
 if attempt:time.sleep(5 if attempt==1 else 15)
 r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=90)
 with (b/'dependency-failure-recovery-observation.log').open('a') as f:f.write(json.dumps(dict(attempt=attempt,rc=r.returncode,stdout=r.stdout,stderr=r.stderr))+'\n')
 print(r.returncode,r.stdout,r.stderr,flush=True)
 if not r.returncode:
  (b/'observed-startup-failure.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n');break
