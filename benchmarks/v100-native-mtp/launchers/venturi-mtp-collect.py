import pathlib,subprocess,time
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');out=root/'benchmarks/v100-native-mtp/v100'
ssh='ssh -o BatchMode=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o ProxyCommand=none -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
for name,src in [('validation-v1','v100-mtp-validation-v1'),('export-v1','v100-mtp-export-v1'),('qualification-v2','v100-mtp-qualification-v2'),('qualification-v3','v100-mtp-qualification-v3'),('final-v4','v100-mtp-final-v4'),('diagnostics-v1','v100-mtp-diagnostics-v1'),('build-control','v100-mtp-build/1402b93ce328ef77'),('build-final','v100-mtp-build/fb11e6eff12b2fbf')]:
 dest=out/name;dest.mkdir(exist_ok=True)
 cmd=['rsync','-az','--exclude=*.pt','--exclude=*.so','--exclude=*.sqlite','--exclude=*.sqlite-*','--exclude=*.nsys-rep','--exclude=*.qdstrm','--exclude=.build.lock','-e',ssh,'WZU_Server:/home/wzu/codex-build/Venturi/'+src+'/',str(dest)+'/']
 for attempt in range(3):
  r=subprocess.run(cmd,timeout=180)
  if r.returncode==0:break
  if attempt==2:r.check_returncode()
  time.sleep(2**attempt)
 print(name,flush=True)
