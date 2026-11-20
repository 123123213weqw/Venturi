"""Matched 512-token context, native C8 exact-state test; no quality claim."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import torch
from venturi.models import NativeQwenModel,QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend
from benchmark_model import signature,measured
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--mode',choices=['static','vram-first'],required=True);p.add_argument('--reference');p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=False)
artifact='/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.8-27b-text-q8-rms-v1.ninfer'
base=json.loads((Path(__file__).parent/'pinned-prompts.json').read_text())['prompt_token_ids'];prompts=[(x*32)[:512] for x in base];assert all(len(x)==512 for x in prompts)
w=QwenQ8Weights.from_ninfer(NInferArtifact(artifact));torch.set_num_threads(4)
m=NativeQwenModel.from_q8(w,CudaBackend(a.library),weight_mode='resident',dtype=torch.float16,max_tokens=32,reserve_bytes=256*2**20,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',projection_profile='q8-residual-column-mlp-head-gdn',kv_cache=a.mode,kv_gpu_budget_bytes=128*2**20 if a.mode!='static' else 0,kv_ram_budget_bytes=4096*2**20 if a.mode!='static' else 0)
b=m.create_batch([1024]*8)
try:
 logits,prefill=measured(lambda:b.forward(prompts),4096)
 payload=dict(logits=signature(logits),prefill_states=[signature(b.snapshot(i)) for i in range(8)])
 # Feed the chosen next token as continuation; all runs do exactly the same work.
 ids=logits.argmax(-1).tolist();before=m.kv_cache_info()
 result,decode=measured(lambda:b.generate([[x] for x in ids],max_new_tokens=8,eos_token_ids=[]),64)
 after=m.kv_cache_info();payload.update(tokens=result,committed_states=[signature(b.snapshot(i)) for i in range(8)])
 r=dict(completed=True,profiled=False,mode=a.mode,artifact_id=w.artifact.artifact_id,prompt_tokens=512,output_tokens=8,batch=8,capacity=1024,prefill=prefill,decode=decode,kv_before_decode=before,kv_after_decode=after,payload=payload,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 if a.reference:assert payload==json.loads(Path(a.reference).read_text())['payload'];r['exact_reference_bytes_equal']=True
 (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
finally:b.close();m.close()
