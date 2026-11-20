import subprocess,os,signal,json,hashlib,time,tarfile,io
from pathlib import Path
R=Path(__file__).resolve().parents[2];C=R/'benchmarks/v100-five-experiments';base='/home/data/wangyue/repos/codex-build/Venturi'
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=8','-o','ServerAliveCountMax=2']
if os.environ.get('USE_IPV6'):
 ssh+=['-6','-o','HostName=fd7a:115c:a1e0::5e35:bc73','-o','HostKeyAlias=[100.70.188.115]:9022','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/venturi-five-ipv6-%r-%h-%p']
files={n:R/n for n in ['src/venturi/qwen_export_verify.py','tests/test_native_experiments.py','src/venturi/ops/csrc/qwen_runtime.cuh','src/venturi/ops/csrc/qwen_head_gdn.cuh','docs/v100-five-experiments.md']}
control=['run-window-v4.py','build-worker-v2.py','benchmark_model.py','benchmark_mtp.py','test_native_experiments_extra.py','pinned-prompts.json']
for n in control:files['control/'+n]=C/n
files['control/restore_service_v1.py']=C/'restore_service_v4.py'
pins={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools','docs'] for p in (R/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
pins.update({f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['AGENTS.md','pyproject.toml','README.md']})
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as tf:
 for name,p in files.items():tf.add(p,arcname=name)
 value=json.dumps(pins).encode();info=tarfile.TarInfo('local-source-pins.json');info.size=len(value);tf.addfile(info,io.BytesIO(value))
data=b.getvalue();local=C/'source-control-delta-v4.tgz';local.write_bytes(data);digest=hashlib.sha256(data).hexdigest()
remote=base+'/five-delta-v4.tgz'
command=f'cat > {remote}; test "$(sha256sum {remote} | cut -d" " -f1)" = {digest} && mkdir -p {base}/five-src-v4 {base}/five-control-v4 && cp -aln {base}/five-src-v3/. {base}/five-src-v4/ && rm -f {base}/five-src-v4/src/venturi/qwen_export_verify.py {base}/five-src-v4/tests/test_native_experiments.py {base}/five-src-v4/src/venturi/ops/csrc/qwen_runtime.cuh {base}/five-src-v4/src/venturi/ops/csrc/qwen_head_gdn.cuh && tar xzf {remote} -C {base}/five-src-v4 && cp -a {base}/five-src-v4/control/. {base}/five-control-v4/ && echo TRANSFER_COMPLETE'
t=time.time();p=subprocess.Popen(ssh+['WZU_Server',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
try:o,e=p.communicate(data,timeout=65);rc=p.returncode
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGKILL)
 for f in [p.stdin,p.stdout,p.stderr]:f.close()
 p.wait(timeout=5);o,e,rc=b'',b'outer timeout',124
j=dict(rc=rc,stdout=o.decode(errors='replace'),stderr=e.decode(errors='replace'),seconds=time.time()-t,bytes=len(data),sha256=digest,route='ipv6' if os.environ.get('USE_IPV6') else 'primary')
(C/('recover-transfer-v4-'+j['route']+'.json')).write_text(json.dumps(j,indent=2));print(json.dumps(j),flush=True)
