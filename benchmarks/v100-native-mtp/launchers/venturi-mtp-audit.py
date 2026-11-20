import subprocess
code=r'''
import pathlib,hashlib,json,subprocess,time
root=pathlib.Path('/home/wzu/codex-build/Venturi');src=root/'v100-mtp-src-v4';out=root/'v100-mtp-diagnostics-v1'
lib=pathlib.Path((root/'v100-mtp-final-v4/library.txt').read_text().strip())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(time_unix=time.time(),library=str(lib),library_sha256=sha(lib),source_sha256={str(p.relative_to(src)):sha(p) for base in ['src','tests','tools'] for p in (src/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts},owned_task_processes=[s for s in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines() if 'v100-mtp' in s and 'sshd:' not in s])
(out/'executed-source.json').write_text(json.dumps(r,indent=2)+'\n')
retained=[]
for directory in ['v100-mtp-qualification-v2','v100-mtp-diagnostics-v1']:
 for p in (root/directory).rglob('*'):
  if p.is_file() and p.suffix in ['.pt','.sqlite','.nsys-rep']:
   retained.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
(out/'remote-retained-files.json').write_text(json.dumps(retained,indent=2)+'\n')
print(json.dumps(dict(native_source_files=len(r['source_sha256']),library_sha256=r['library_sha256'],retained_files=len(retained))))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
