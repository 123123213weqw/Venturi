import pathlib,json,torch
from venturi.models import QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend,prepare_q8_dp4a
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-dp4a-final-v4';lib=(root/'artifacts/venturi-dp4a-vector-v3/library.txt').read_text().strip();torch.set_num_threads(4)
b=CudaBackend(lib);w=QwenQ8Weights.from_ninfer(NInferArtifact(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'));spec=next(s for s in w.plan.bindings if s.shape==(17408,5120) and s.format=='q8_g32_fp16');r=b.upload_linear(w.read_linear(spec.binding),dtype=torch.float16,device='cuda:0')
for t in [1,8]:
 x=torch.randn(t,5120,device='cuda:0',dtype=torch.float16,generator=torch.Generator(device='cuda').manual_seed(719+t));plans=[r.prepare(tokens=t)]+[prepare_q8_dp4a(r,tokens=t,residual_compensation=p) for p in [False,True]];scratch=[None]+[torch.empty(p.workspace_bytes,dtype=torch.uint8,device='cuda:0') for p in plans[1:]];outs=[torch.empty(t,17408,dtype=torch.float16,device='cuda:0') for _ in plans]
 for i in range(3):
  if i:plans[i].run(x,out=outs[i],workspace=scratch[i])
  else:plans[i].run(x,out=outs[i])
 torch.cuda.synchronize()
 for i in range(3):
  with torch.cuda.nvtx.range(f'projection_{i}_t{t}'):
   for _ in range(16):
    if i:plans[i].run(x,out=outs[i],workspace=scratch[i])
    else:plans[i].run(x,out=outs[i])
  torch.cuda.synchronize()
(out/'profile-scope.json').write_text(json.dumps(dict(profiled=True,scope='Standalone projection attribution, not a full-model/serving run or throughput. FP16 activation baseline versus explicit A8/F32scale one pass and residual two pass. GPU work is joined by launch correlation IDs; asynchronous kernels may finish beyond CPU NVTX ends.',shape=[17408,5120],rows=[1,8],calls_per_range=16,range_profiles={'0':'FP16 activation original Q8','1':'A8 one pass','2':'A8 residual compensation'},library=lib),indent=2)+'\n')
