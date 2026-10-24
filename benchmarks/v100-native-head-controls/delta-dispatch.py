from pathlib import Path
import ast,base64,difflib,hashlib,json,subprocess,time,zlib
b=Path('benchmarks/v100-native-head-controls')
def extract_worker(p):
 tree=ast.parse(p.read_text());code=next(ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and any(getattr(t,'id','')=='code' for t in x.targets));inner=ast.parse(code)
 return next(ast.literal_eval(x.args[0]) for x in ast.walk(inner) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='write_text' and x.args and isinstance(x.args[0],ast.Constant) and isinstance(x.args[0].value,str) and len(x.args[0].value)>10000)
def pack(old,new):
 ops=[]
 for tag,i,j,a,c in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
  if tag=='equal':ops.append([i,j])
  elif tag in ['insert','replace']:ops.append(new[a:c])
 assert ''.join(old[x[0]:x[1]] if isinstance(x,list) else x for x in ops)==new
 raw=json.dumps(ops,separators=(',',':')).encode();return dict(old_sha=hashlib.sha256(old.encode()).hexdigest(),new_sha=hashlib.sha256(new.encode()).hexdigest(),payload=base64.b64encode(zlib.compress(raw,9)).decode(),new_bytes=len(new.encode()),operations=len(ops))
old4=extract_worker(b/'serialized-dispatch-v1.py');new4=extract_worker(b/'serialized-recovery-v4-dispatch.py');assert new4==(b/'prepared-recovery-v4-worker.py').read_text()
old5=extract_worker(Path('/tmp/venturi-native-head-aligned-v2-dispatch.py'));new5=extract_worker(b/'serialized-aligned-v5-dispatch.py');assert new5==(b/'prepared-aligned-v5-worker.py').read_text()
data={'v4':pack(old4,new4),'v5':pack(old5,new5)}
wire="""import pathlib,json,hashlib,zlib,base64,subprocess,time,socket
assert socket.gethostname()=='wzu-SYS-4029GP-TRT'
base=pathlib.Path('/home/data/wangyue/artifacts')
def state(out):
 d={'artifact_exists':out.exists()}
 for f in ['process.json','exit-code.txt','failure.txt']:
  if (out/f).exists():d[f]=(out/f).read_text()[-2000:]
 if (out/'process.json').exists():
  p=pathlib.Path('/proc')/str(json.loads((out/'process.json').read_text())['pid'])/'cmdline';d['live_cmdline']=p.read_bytes().decode(errors='replace') if p.exists() else None
 return d
def deploy(old_name,new_name,pin):
 out=base/new_name
 if not out.exists():
  old=(base/old_name/'worker.py').read_text();assert hashlib.sha256(old.encode()).hexdigest()==pin['old_sha'],'Existing executed worker differs; never improvise'
  ops=json.loads(zlib.decompress(base64.b64decode(pin['payload'])));source=''.join(old[x[0]:x[1]] if isinstance(x,list) else x for x in ops);assert hashlib.sha256(source.encode()).hexdigest()==pin['new_sha']
  compile(source,str(out/'worker.py'),'exec')
  try:out.mkdir()
  except FileExistsError:pass
  else:
   p=out/'worker.py';p.write_text(source)
   with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
   (out/'process.json').write_text(json.dumps({'pid':proc.pid,'started_unix':time.time()},indent=2))
 r=state(out);print(json.dumps({new_name:r}),flush=True);return r
r=deploy('venturi-native-head-controls-v1','venturi-native-head-controls-v4',DATA['v4'])
if r.get('live_cmdline') and 'venturi-native-head-controls-v4/worker.py' in r['live_cmdline'] or r.get('exit-code.txt','').strip()=='0':deploy('venturi-native-head-aligned-v2','venturi-native-head-aligned-v5',DATA['v5'])
""".replace('DATA',repr(data))
(b/'delta-v4-v5-observe-and-dispatch.py').write_text(wire);(b/'delta-dispatch-pins.json').write_text(json.dumps({k:{x:y for x,y in v.items() if x!='payload'} for k,v in data.items()},indent=2)+'\n');print('Locally reconstructed exact preserved workers, wire bytes',len(wire.encode()),flush=True)
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
for attempt,route in enumerate(['direct','master']):
 if attempt:time.sleep(30)
 ssh=n['SSH'][:];ssh[ssh.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6'
 if route=='master':ssh[ssh.index('ControlPath=none')]='ControlPath=/Users/wangyue/.ssh/cm-14974355c3f315e78f58843cae0595e186514110'
 try:r=subprocess.run(ssh,input=wire,text=True,capture_output=True,timeout=70)
 except subprocess.TimeoutExpired as e:
  with (b/'delta-dispatch.log').open('a') as f:f.write(json.dumps(dict(route=route,timeout=str(e)))+'\n')
  print(route,'timeout; future request must inspect first',flush=True);continue
 result=dict(route=route,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
 with (b/'delta-dispatch.log').open('a') as f:f.write(json.dumps(result)+'\n')
 print(result,flush=True)
 if not r.returncode:
  (b/'delta-dispatch-result.json').write_text(json.dumps(result,indent=2)+'\n');(b/'delta-healthy-route.json').write_text(json.dumps(dict(ssh=ssh),indent=2)+'\n');break
