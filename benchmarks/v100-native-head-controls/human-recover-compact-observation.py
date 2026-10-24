import pathlib,json
b=pathlib.Path('/home/data/wangyue/artifacts')
for n in ['venturi-native-head-controls-v13','venturi-native-head-aligned-v14','venturi-native-head-controls-profile-v15']:
 p=b/n;d={'name':n};pid=json.loads((p/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');d['cmd']=q.read_bytes().decode() if q.exists() else None
 for f in ['exit-code.txt','failure.txt','completed.json']:
  if (p/f).exists():d[f]=(p/f).read_text()[-1600:]
 if (p/'campaign-report.json').exists():
  r=json.loads((p/'campaign-report.json').read_text());d['experiments']=[(e['label'],e['completed']) for e in r['experiments']]
 print(json.dumps(d),flush=True)
