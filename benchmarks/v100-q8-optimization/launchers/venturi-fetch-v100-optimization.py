import subprocess,pathlib,tarfile,io
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi/benchmarks/v100-q8-optimization')
opts=['-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
cmd='tar -czf - --exclude="*.pt" --exclude="*.ninfer" --exclude="*.safetensors" --exclude="*.sqlite" --exclude="*.nsys-rep" --exclude="*.qdstrm" -C /home/wzu/codex-build/Venturi/v100-resident-optimization-v1 .'
r=subprocess.run(['ssh']+opts+['WZU_Server',cmd],capture_output=True,timeout=120);r.check_returncode()
dst=root/'v100';dst.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tf:
 for m in tf.getmembers():
  p=pathlib.PurePosixPath(m.name)
  assert not p.is_absolute() and '..' not in p.parts
  if m.isdir():continue
  assert m.isfile()
  target=dst/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
print('Fetched',len(r.stdout),'compressed bytes')
