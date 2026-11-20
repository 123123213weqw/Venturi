import pathlib,subprocess,json
code='''import pathlib,subprocess,json,time
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
assert (p/'baseline-exit-code.txt').read_text().strip()=='0'
assert (p/'tiling-build-exit-code.txt').read_text().strip()=='0'
assert not (p/'tiling-tests-process.json').exists()
with (p/'tiling-tests-worker.log').open('xb') as log:
 proc=subprocess.Popen(['bash',str(p/'tiling-tests-worker.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(p/'tiling-tests-process.json').write_text(json.dumps(r));print(json.dumps(r))
'''
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
subprocess.run(['scp']+opts+['/tmp/venturi-tiling-tests-worker.sh','WZU_Server:/home/wzu/codex-build/Venturi/v100-resident-optimization-v1/tiling-tests-worker.sh'],check=True,timeout=60)
r=subprocess.run(['ssh']+opts+['WZU_Server','python3 -'],input=code,text=True,capture_output=True,timeout=60)
print(r.stdout,r.stderr);r.check_returncode()
