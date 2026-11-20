from pathlib import Path
import subprocess,json,ast,hashlib,zlib,base64,time
b=Path('benchmarks/v100-native-head-controls');n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
tree=ast.parse((b/'serialized-recovery-v4-dispatch.py').read_text());code=next(ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and any(getattr(t,'id','')=='code' for t in x.targets));raw=code.encode();encoded=base64.b64encode(zlib.compress(raw,9)).decode();pin=hashlib.sha256(raw).hexdigest()
wire="""import pathlib,json,socket,base64,zlib,hashlib
assert socket.gethostname()=='wzu-SYS-4029GP-TRT','Never switch the benchmark host'
p=pathlib.Path('/home/data/wangyue/artifacts/venturi-native-head-controls-v4')
if p.exists():
 r={'existing_artifact':True}
 for f in ['process.json','exit-code.txt','failure.txt']:
  if (p/f).exists():r[f]=(p/f).read_text()[-2000:]
 if (p/'process.json').exists():
  q=pathlib.Path('/proc')/str(json.loads((p/'process.json').read_text())['pid'])/'cmdline';r['live_cmdline']=q.read_bytes().decode(errors='replace') if q.exists() else None
 print(json.dumps(r),flush=True)
else:
 print(json.dumps({'observed_artifact_absent':True,'action':'exact preserved dispatch in this authenticated connection'}),flush=True)
 s=zlib.decompress(base64.b64decode(PAYLOAD));assert hashlib.sha256(s).hexdigest()==PIN
 exec(compile(s,'preserved-v4-dispatch','exec'))
""".replace('PAYLOAD',repr(encoded)).replace('PIN',repr(pin));(b/'guarded-v4-observe-and-dispatch.py').write_text(wire)
for attempt,route in enumerate(['direct','direct','master']):
 if attempt:time.sleep(15 if attempt==1 else 30)
 ssh=n['SSH'][:];ssh[ssh.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6'
 if route=='master':ssh[ssh.index('ControlPath=none')]='ControlPath=/Users/wangyue/.ssh/cm-14974355c3f315e78f58843cae0595e186514110'
 try:r=subprocess.run(ssh,input=wire,text=True,capture_output=True,timeout=65)
 except subprocess.TimeoutExpired as e:
  with (b/'guarded-v4-dispatch.log').open('a') as f:f.write(json.dumps(dict(attempt=attempt,route=route,timeout=str(e)))+'\n')
  print(attempt,route,'observation/dispatch timeout; next attempt first inspects the artifact, never blindly relaunches',flush=True);continue
 data=dict(attempt=attempt,route=route,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
 with (b/'guarded-v4-dispatch.log').open('a') as f:f.write(json.dumps(data)+'\n')
 print(data,flush=True)
 if not r.returncode:
  (b/'guarded-v4-success.json').write_text(json.dumps(dict(ssh=ssh,result=data),indent=2)+'\n');break
