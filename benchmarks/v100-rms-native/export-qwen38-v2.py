"""CPU-only Qwen3.8 acquisition recovery using TLS-verified immutable Hub metadata."""
import hashlib,json,time
from pathlib import Path
import torch
from venturi.checkpoint import inspect_checkpoint
from venturi.models.qwen_source import QwenSourcePlan
from venturi.qwen_export import export_qwen_q8

out=Path('/home/data/wangyue/artifacts/venturi-rms-native-qwen38-export-v2');out.mkdir(exist_ok=False)
control=Path(__file__).parent
source=Path('/home/wzu/models/Qwen3.8-27B')
raw=(control/'hf-source-metadata.json').read_bytes();hub=json.loads(raw)
revision=(source/'.model-revision').read_text().strip();assert hub['sha']==revision=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
retrieval=json.loads((control/'hf-source-metadata-attempt.json').read_text());assert retrieval['status']==200
assert retrieval['url']=='https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/'+revision+'?blobs=true'
checkpoint=inspect_checkpoint(source);plan=QwenSourcePlan.from_config(dict(checkpoint.config));plan.validate_checkpoint(checkpoint)
torch.set_num_threads(4)
def save(n,j):(out/n).write_text(json.dumps(j,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*2**20),b''):h.update(b)
 return h.hexdigest()
required={p.name for p in checkpoint.files}|{'config.json','model.safetensors.index.json','tokenizer.json'}
by_name={f['rfilename']:f for f in hub['siblings']};files=[]
for name in sorted(required):
 p=source/name;m=by_name[name];before=p.stat();assert before.st_size==m['size'],name
 digest=sha(p);after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 lfs=m.get('lfs',{}).get('sha256')
 if lfs:assert digest==lfs,name
 else:
  raw=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==m['blobId'],name
 files.append(dict(name=name,bytes=before.st_size,actual_sha256=digest,lfs_sha256=lfs,git_blob_id=m['blobId']))
 save('source-pin-progress.json',dict(revision=revision,files=files));print('verified',name,flush=True)
manifest=dict(status='complete',repo='Qwen/Qwen3.8-27B',revision=revision,source_endpoint='https://huggingface.co',files=files,
 metadata_sha256=hashlib.sha256((control/'hf-source-metadata.json').read_bytes()).hexdigest(),retrieval=retrieval,
 verification_scope='All source/config/index/tokenizer bytes match official immutable-revision Hub API LFS SHA256 or Git blob ID; metadata fetched over verified TLS on client, hash-pinned transfer to same V100 host.')
save('source-evidence.json',manifest)
artifact=Path('/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.8-27b-text-q8-rms-v1.ninfer')
report=export_qwen_q8(source,artifact,source_manifest=out/'source-evidence.json',rows_per_chunk=256)
save('export-report.json',report);save('completed.json',dict(completed=True,artifact=str(artifact),revision=revision,ended_unix=time.time()))
