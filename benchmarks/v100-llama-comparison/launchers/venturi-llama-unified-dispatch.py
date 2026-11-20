import subprocess
code=r'''
import pathlib,subprocess,json,time
base=pathlib.Path('/home/data/wangyue/artifacts/venturi-llama-comparison-v1');out=base/'unified-recovery-v3';out.mkdir()
assert (base/'pair-screen-v1/exit-code.txt').read_text().strip()=='0'
s=(base/'recovery-v2/worker.py').read_text().replace("out=base/'recovery-v2'", "out=base/'unified-recovery-v3'")
a=s.index(' configurations=[');b=s.index('\n for label,engine,mode,depth',a)
s=s[:a]+" configurations=[('llama-tensor-mtp1-unified','llama','tensor',1)]"+s[b:]
s=s.replace("'--parallel','8','-b'", "'--parallel','8','--kv-unified','-b'")
s=s.replace("except BaseException:item['failure']=traceback.format_exc();", "except BaseException:item['server_returncode_at_failure']=server.poll() if server else None;item['failure']=traceback.format_exc();")
s=s.replace("   stop(server)\n", "   stop(server)\n   item['server_final_returncode']=server.returncode if server else None\n")
s=s.replace('Recovered unprofiled screen:', 'Bounded C8 transport/engine diagnosis with explicitly unified KV:')
(out/'worker.py').write_text(s)
with (out/'worker.log').open('xb') as log:p=subprocess.Popen(['python3',str(out/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=p.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
