from pathlib import Path
import json,hashlib,subprocess,base64
root=Path.cwd();out=root/'benchmarks/v100-activation-words';t=json.loads((out/'terminal-audit.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();need={}
for snap,files in t['executed_snapshots'].items():
 need[snap]={rel:h for rel,h in files.items() if not (root/rel).is_file() or sha(root/rel)!=h}
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
code='import pathlib,json,hashlib,base64\nr=pathlib.Path("/home/data/wangyue/repos/codex-build/Venturi");needs='+repr(need)+'\nout={}\nfor snap,files in needs.items():\n out[snap]={}\n for rel,h in files.items():\n  data=(r/snap/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==h;out[snap][rel]=base64.b64encode(data).decode()\nprint(json.dumps(out))\n'
r=subprocess.run(ns['SSH'],input=code,text=True,capture_output=True,timeout=90);r.check_returncode();d=json.loads(r.stdout)
for snap,files in d.items():
 for rel,value in files.items():
  p=out/'executed-source-deltas'/snap/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value));assert sha(p)==need[snap][rel]
print('Executed source deltas preserved:',{k:len(v) for k,v in need.items()})
