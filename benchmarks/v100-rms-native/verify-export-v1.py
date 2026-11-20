"""Independent source/storage qualification before the approved model window."""
import hashlib,json,time
from pathlib import Path
import torch
from safetensors import safe_open
from venturi.checkpoint import inspect_checkpoint
from venturi.models.qwen_source import QwenSourcePlan
from venturi.models.qwen_weights import QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.qwen_export import _digest,_stamp
out=Path('/home/data/wangyue/artifacts/venturi-rms-native-qwen38-storage-v1');out.mkdir(exist_ok=False)
export=Path('/home/data/wangyue/artifacts/venturi-rms-native-qwen38-export-v2')
for _ in range(600):
 if (export/'completed.json').exists():break
 time.sleep(2)
else:raise TimeoutError('CPU export not complete; no production mutation')
source=Path('/home/wzu/models/Qwen3.8-27B');report=json.loads((export/'export-report.json').read_text());target=Path(report['output']);before=_stamp(target)
assert _digest(target)==report['sha256']
checkpoint=inspect_checkpoint(source);plan=QwenSourcePlan.from_config(dict(checkpoint.config));plan.validate_checkpoint(checkpoint)
container=NInferArtifact(target);weights=QwenQ8Weights.from_ninfer(container);assert weights.config==plan.config and container.artifact_id==report['artifact_id']
floats=rows=bf16_gdn=0;maximum=0.
torch.set_num_threads(4)
for spec in plan.bindings:
 obj=container.objects[container.manifest['bindings'][spec.binding]['object']]
 with safe_open(str(checkpoint.tensors[spec.source].file),framework='pt',device='cpu') as r:
  if spec.format=='q8_g32_fp16':
   n,k=spec.shape;s=r.get_slice(spec.source)
   for row in sorted({0,n//2,n-1}):
    raw=s[row:row+1].double().flatten().tolist();expected=[]
    for start in range(0,k,32):
     group=raw[start:start+32];scale=torch.tensor(max(map(abs,group))/127,dtype=torch.float16).item()
     expected.extend(min(127,max(-127,round(x/scale)))*scale if scale else 0 for x in group)
    actual=container.decode_row(obj['id'],row);assert actual==expected,(spec.binding,row)
    maximum=max(maximum,max(abs(a-b) for a,b in zip(actual,raw)));rows+=1
  else:
   value=r.get_tensor(spec.source)
   if spec.transform=='conv-kc':value=value.squeeze(1).transpose(0,1)
   dtype=torch.float32 if spec.format=='fp32' else torch.bfloat16
   actual=torch.frombuffer(bytearray(container.read_object(obj['id'],max_bytes=8*2**20)),dtype=dtype).reshape(spec.shape)
   assert torch.equal(actual,value.to(dtype)),spec.binding;floats+=1
   if spec.binding.endswith('/gdn/norm') and value.dtype==torch.bfloat16:bf16_gdn+=1
assert _stamp(target)==before and bf16_gdn==48
j=dict(completed=True,artifact_id=container.artifact_id,artifact_sha256=report['sha256'],full_floating_tensors_equal=floats,independent_scalar_q8_rows_equal=rows,bf16_gdn_norms_exactly_widened=bf16_gdn,max_sampled_abs_weight_quantization_error=maximum,
 scope='All floating payload bytes vs BF16/F32 source; independent scalar nearest-even quantizer on 3 rows per matrix. Not a BF16-vs-Q8 output quality evaluation.',ended_unix=time.time())
(out/'completed.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j),flush=True)
