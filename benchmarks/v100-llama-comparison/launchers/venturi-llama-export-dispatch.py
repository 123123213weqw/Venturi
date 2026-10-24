import subprocess
worker=r'''import pathlib,subprocess,json,os,time,hashlib,traceback,signal
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-llama-comparison-v1';models=root/'models/venturi-qwen35-comparison-v1'
converter=root/'repos/codex-build/llama.cpp-meta-tp-candidate-20260829';source=root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16';py=root/'dependencies/venvs/v100-active/bin/python'
env=os.environ.copy();env.update(PYTHONPATH=str(root/'repos/codex-build/Venturi/v100-mtp-src-v4/src')+':'+str(converter/'gguf-py'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*2**20),b''):h.update(b)
 return h.hexdigest()
def run(cmd,path,timeout=2400):
 with path.open('w') as log:
  p=subprocess.Popen(cmd,cwd=converter,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  if rc:raise RuntimeError(f'Owned CPU export command exited {rc}; inspect {path}')
try:
 models.mkdir();r=dict(started_unix=time.time(),completed=False,source=str(source),converter=str(converter),cuda_visible_devices='',scope='Same original checkpoint, standard llama.cpp Q8_0 export including MTP; not identical quantization to Venturi Q8',model=str(models/'qwen3.5-27b-original-q8_0.gguf'))
 (out/'export-report.json').write_text(json.dumps(r,indent=2))
 r['converter_git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=converter,text=True).strip()
 r['converter_source_sha256']={str(p.relative_to(converter)):digest(p) for base in ['conversion','gguf-py/gguf'] for p in (converter/base).rglob('*.py')}
 r['converter_source_sha256']['convert_hf_to_gguf.py']=digest(converter/'convert_hf_to_gguf.py')
 validate='from venturi.checkpoint import inspect_checkpoint; from venturi.qwen_export import _source_evidence; import pathlib,json; c=inspect_checkpoint('+repr(str(source))+'); evidence,stamps=_source_evidence(c,manifest=pathlib.Path('+repr(str(source/'venturi-source.json'))+'),allow_unverified=False); print(json.dumps(dict(source=evidence,stamps=stamps)))'
 run([str(py),'-c',validate],out/'source-validation.json',timeout=900)
 valid=json.loads((out/'source-validation.json').read_text());r['verified_source']=valid['source'];r['source_stamps_before']=valid['stamps']
 pending=models/'pending-q8_0.gguf';cmd=[str(py),'convert_hf_to_gguf.py',str(source),'--outtype','q8_0','--outfile',str(pending)]
 r['command']=cmd;(out/'export-report.json').write_text(json.dumps(r,indent=2))
 run(cmd,out/'convert.log')
 for name,stamp in r['source_stamps_before'].items():
  s=(source/name).stat();assert list((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))==stamp,'Original source changed during conversion'
 inspect='import gguf,json; from collections import Counter; r=gguf.GGUFReader('+repr(str(pending))+'); print(json.dumps(dict(tensor_count=len(r.tensors),tensor_types=dict(Counter(str(t.tensor_type) for t in r.tensors)),mtp_tensor_names=[t.name for t in r.tensors if t.name.startswith("blk.64.")],architecture=str(r.fields["general.architecture"].contents()))))'
 run([str(py),'-c',inspect],out/'gguf-inventory.json',timeout=120)
 inv=json.loads((out/'gguf-inventory.json').read_text());assert len(inv['mtp_tensor_names'])>=15 and inv['architecture']=='qwen35',inv
 final=pathlib.Path(r['model']);assert not final.exists();pending.rename(final)
 r.update(completed=True,ended_unix=time.time(),gguf_sha256=digest(final),gguf_bytes=final.stat().st_size,inventory=inv)
 (out/'export-report.json').write_text(json.dumps(r,indent=2)+'\n');rc=0
except BaseException:(out/'export-failure.txt').write_text(traceback.format_exc());rc=1
(out/'export-exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
'''
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-llama-comparison-v1');out.mkdir();(out/'export-worker.py').write_text({worker!r})
with (out/'export-worker.log').open('xb') as log:p=subprocess.Popen(['python3',str(out/'export-worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=p.pid,started_unix=time.time());(out/'export-process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
