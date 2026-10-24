import subprocess
code=r'''
import pathlib,subprocess,time,json
base=pathlib.Path('/home/data/wangyue/artifacts/venturi-llama-comparison-v1');out=base/'recovery-v2';out.mkdir()
s=(base/'screen-worker.py').read_text()
s=s.replace("out=root/'artifacts/venturi-llama-comparison-v1';src=", "base=root/'artifacts/venturi-llama-comparison-v1';out=base/'recovery-v2';src=")
s=s.replace("llama-compare-src-v1'","llama-compare-src-v2'")
s=s.replace("terminal=out/'export-exit-code.txt'","terminal=base/'export-exit-code.txt'")
s=s.replace("(out/'export-report.json')","(base/'export-report.json')")
a=s.index(' configurations=[');b=s.index('\n for label,engine,mode,depth',a)
s=s[:a]+" configurations=[('llama-tensor-off','llama','tensor',0),('llama-tensor-mtp1','llama','tensor',1),('llama-tensor-mtp3','llama','tensor',3),('llama-layer-off','llama','layer',0),('llama-layer-mtp1','llama','layer',1),('llama-layer-mtp3','llama','layer',3),('venturi-tp-off','venturi','tp',0)]"+s[b:]
s=s.replace(" save();terminal=", " for _ in range(180):\n  if (base/'screen-exit-code.txt').exists():break\n  time.sleep(10)\n else:raise RuntimeError('Bounded initial-worker wait elapsed; no recovery GPU process started')\n save();terminal=")
s=s.replace('Initial unprofiled loopback HTTP screen, same original checkpoint and fixed work.', 'Recovered unprofiled screen: prompt_n gates actual prompt work; tokens_cached is retained KV, not reused input. Prior failed attempt preserved.')
(out/'worker.py').write_text(s)
with (out/'worker.log').open('xb') as log:p=subprocess.Popen(['python3',str(out/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=p.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
