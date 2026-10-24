from pathlib import Path
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
s=Path('/tmp/venturi-native-column-mlp-profile-v7-worker.py').read_text().replace('venturi-native-column-mlp-profile-v7','venturi-native-column-mlp-profile-v9')
s=s.replace("'--projection-rows','256','--projection-profile'","'--projection-rows','256','--mode','batch','--projection-profile'")
a="   prefix=out/name;cmd+=['--output',str(out/(name+'-metadata.json'))]"
b=r'''   previous=root/'artifacts/venturi-native-column-mlp-profile-v7'
   if name in ['same-library-row-mtp2-c1','column-mtp2-c1']:
    assert (previous/'exit-code.txt').read_text().strip()=='1'
    import shutil
    for suffix in ['-metadata.json','-attribution.json','-profile.log','-export.log','-attribution.log']:
     shutil.copyfile(previous/(name+suffix),out/(name+suffix))
    traces+=['--trace',name+'='+str(previous/(name+'.sqlite'))]
    pin=out/(name+'-reused-trace.json');pin.write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [previous/(name+'.sqlite'),previous/(name+'.nsys-rep')]},indent=2)+chr(10))
    continue
   prefix=out/name;cmd+=['--output',str(out/(name+'-metadata.json'))]'''
assert a in s;s=s.replace(a,b)
ns['dispatch']('venturi-native-column-mlp-profile-v9',s)
cli=Path('/tmp/venturi-native-column-mlp-cli-v8-worker.py').read_text().replace('venturi-native-column-mlp-cli-v8','venturi-native-column-mlp-cli-v10').replace('venturi-native-column-mlp-profile-v7','venturi-native-column-mlp-profile-v9')
ns['dispatch']('venturi-native-column-mlp-cli-v10',cli)
