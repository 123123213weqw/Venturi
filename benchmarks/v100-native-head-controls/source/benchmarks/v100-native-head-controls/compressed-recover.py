from pathlib import Path
import subprocess,json,ast,hashlib,zlib,base64,time
b=Path('benchmarks/v100-native-head-controls');state=json.loads((b/'turn3-authoritative-state.json').read_text());assert state['host']=='wzu-SYS-4029GP-TRT' and not state['venturi-native-head-controls-v4']['exists'] and not state['venturi-native-head-aligned-v5']['exists']
ssh=json.loads((b/'turn3-healthy-route.json').read_text())['ssh'];tree=ast.parse((b/'serialized-recovery-v4-dispatch.py').read_text());code=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='code' for t in n.targets));raw=code.encode();h=hashlib.sha256(raw).hexdigest();encoded=base64.b64encode(zlib.compress(raw,9)).decode()
wire='import zlib,base64,hashlib\ns=zlib.decompress(base64.b64decode('+repr(encoded)+'));assert hashlib.sha256(s).hexdigest()=='+repr(h)+';exec(compile(s,"preserved-v4-dispatch","exec"))\n'
(b/'compressed-recovery-v4-dispatch.py').write_text(wire);(b/'compressed-recovery-v4-metadata.json').write_text(json.dumps(dict(observation_sha256=hashlib.sha256((b/'turn3-authoritative-state.json').read_bytes()).hexdigest(),original_program_sha256=h,program_bytes=len(raw),wire_bytes=len(wire.encode()),no_worker_source_change=True),indent=2)+'\n')
print('Verified absent; exact serialized program compressed',len(raw),'->',len(wire),flush=True)
try:r=subprocess.run(ssh,input=wire,text=True,capture_output=True,timeout=80)
except subprocess.TimeoutExpired as e:(b/'compressed-v4-dispatch.log').write_text(str(e));print('Dispatch timeout; inspect outcome, never blindly replay',flush=True);raise SystemExit(0)
(b/'compressed-v4-dispatch.log').write_text(r.stdout+r.stderr);print(r.returncode,r.stdout,r.stderr,flush=True)
if not r.returncode:(b/'v4-dispatch-result.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
