import pathlib,subprocess
worker=pathlib.Path('/tmp/venturi-strided-campaign-v4-worker.py').read_text()
import textwrap
worker="try:\n"+textwrap.indent(worker," ")+"\nexcept Exception:\n import pathlib,traceback\n out=pathlib.Path('/home/data/wangyue/artifacts/venturi-strided-campaign-v4')\n (out/'failure.txt').write_text(traceback.format_exc())\n (out/'exit-code.txt').write_text('1\n')\n raise\n"
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-strided-campaign-v4');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
