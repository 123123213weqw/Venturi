import hashlib,io,json,tarfile,time
from pathlib import Path
ROOT=Path('/home/data/wangyue/artifacts')
labels={'build-v1':ROOT/'venturi-kv-tier-v1','build-v2':ROOT/'venturi-kv-tier-v2','build-v3':ROOT/'venturi-kv-tier-v3','window-v2':ROOT/'venturi-kv-tier-window-v1','window-final-v3':ROOT/'venturi-kv-tier-window-final-v3','lifetime-v4':ROOT/'venturi-kv-tier-lifetime-v4'}
for label in ['build-v2','build-v3','window-v2','window-final-v3','lifetime-v4']:
 assert json.loads((labels[label]/'completed.json').read_text())['completed'],label
files={}
for label,d in labels.items():
 for p in d.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix not in ['.so','.tmp'] and p.name not in ['.build.lock','kv_tier_probe'] and '__pycache__' not in p.parts:
   files[label+'/'+str(p.relative_to(d))]=p
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(files.items())}
archive=ROOT/'venturi-kv-tier-evidence-v1.tgz'
with tarfile.open(archive,'w:gz',compresslevel=5) as t:
 for n,p in files.items():t.add(p,arcname=n,recursive=False)
 data=(json.dumps(manifest,indent=2)+'\n').encode();i=tarfile.TarInfo('manifest.json');i.size=len(data);t.addfile(i,io.BytesIO(data))
pin=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(files),unix=time.time())
(ROOT/'venturi-kv-tier-evidence-v1.pin.json').write_text(json.dumps(pin,indent=2)+'\n');print(json.dumps(pin))
