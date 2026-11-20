import pathlib,ast,subprocess
p=pathlib.Path('/tmp/venturi-dp4a-fixture-dispatch.py');tree=ast.parse(p.read_text());worker=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='worker' for t in n.targets)))
worker=worker.replace('dp4a-src-v1','dp4a-src-v4').replace('venturi-dp4a-fixtures-v1','venturi-dp4a-final-v4')
tree=ast.parse(pathlib.Path('/tmp/venturi-dp4a-final-dispatch-v4.py').read_text());body=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets)))
a=worker.index(" run([py,'-m','venturi.ops.build'");b=worker.index(" (out/'completed.json')",a);worker=worker[:a]+body+worker[b:]
payload=pathlib.Path('/tmp/venturi-dp4a-profile-payload.py').read_text();summary=pathlib.Path('/tmp/venturi-dp4a-profile-summary.py').read_text();i=worker.index("\ntry:\n assert (root/");worker=worker[:i]+worker[i:].replace('\ntry:\n',"\ntry:\n (out/'profile_payload.py').write_text("+repr(payload)+");(out/'profile_summary.py').write_text("+repr(summary)+")\n",1)
compile(worker,'worker.py','exec')
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-dp4a-final-v4');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
compile(code,'envelope.py','exec')
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -'];r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
