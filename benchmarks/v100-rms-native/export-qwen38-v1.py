"""CPU-only Qwen3.8 source verification/export; never stops production or loads CUDA."""
import hashlib,json,os,time
from pathlib import Path
import torch
from venturi.checkpoint import inspect_checkpoint
from venturi.models.qwen_source import QwenSourcePlan
from venturi.qwen_export import export_qwen_q8

out=Path('/home/data/wangyue/artifacts/venturi-rms-native-qwen38-export-v1');out.mkdir(exist_ok=False)
source=Path('/home/wzu/models/Qwen3.8-27B')
checkpoint=inspect_checkpoint(source);plan=QwenSourcePlan.from_config(dict(checkpoint.config));plan.validate_checkpoint(checkpoint)
revision=(source/'.model-revision').read_text().strip();assert len(revision)==40
torch.set_num_threads(4)
def save(n,j):(out/n).write_text(json.dumps(j,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*2**20),b''):h.update(b)
 return h.hexdigest()
required={p.name for p in checkpoint.files}|{'config.json','model.safetensors.index.json'}
files=[]
for name in sorted(required):
 p=source/name;meta=source/'.cache/huggingface/download'/(name+'.metadata')
 assert meta.is_file(),f'Acquisition metadata missing; do not claim upstream verification: {name}'
 lines=meta.read_text().splitlines();assert lines[0]==revision,(name,lines[0],revision)
 expected=lines[1].strip('"');before=p.stat();digest=sha(p);after=p.stat()
 assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 if len(expected)==64:assert digest==expected,name
 elif len(expected)==40:
  raw=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected,name
 else:raise ValueError('Unknown acquisition ETag format')
 files.append(dict(name=name,bytes=before.st_size,actual_sha256=digest,lfs_sha256=expected if len(expected)==64 else None,git_blob_id=expected if len(expected)==40 else None))
 save('source-pin-progress.json',dict(revision=revision,files=files));print('verified',name,flush=True)
manifest=dict(status='complete',repo='Qwen/Qwen3.8-27B',revision=revision,source_endpoint='https://huggingface.co',files=files,
 verification_scope='Current complete file hashes match pinned HF download-cache LFS/Git ETags at recorded revision; no new upstream fetch. Config/index Git-blob IDs checked.')
save('source-evidence.json',manifest)
artifact=Path('/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.8-27b-text-q8-rms-v1.ninfer')
report=export_qwen_q8(source,artifact,source_manifest=out/'source-evidence.json',rows_per_chunk=256)
save('export-report.json',report);save('completed.json',dict(completed=True,artifact=str(artifact),revision=revision,ended_unix=time.time()))
