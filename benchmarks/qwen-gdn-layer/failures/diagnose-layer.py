import tempfile,pathlib,json,os
import torch
from test_qwen_gdn_layer import fixture_weights,inputs
from qwen_gdn_reference import gdn_layer_oracle
from venturi.models import ResidentQwenGDNLayer
from venturi.ops import CudaBackend,TorchBackend

torch.set_num_threads(4)
root=pathlib.Path('/home/test/wangyue/venturi-work/qwen/qwen-gdn-layer-validation')
report=[]
with tempfile.TemporaryDirectory() as d:
 weights,raw,cfg=fixture_weights(pathlib.Path(d),hv=6)
 for name,backend,device in [('cpu-bf16',TorchBackend(),'cpu'),('cuda',CudaBackend('/home/test/wangyue/venturi-work/qwen/own-blocks-build/afd14a02433a8f9c/libventuri_blocks.so'),'cuda:0')]:
  r=ResidentQwenGDNLayer.from_q8(weights,backend,layer=0,dtype=torch.bfloat16,device=device)
  x,s=inputs(r,9);p=r.prepare(tokens=9);w=p.empty_workspace();o=torch.empty_like(x);h,st=torch.empty_like(s.history),torch.empty_like(s.recurrent)
  ideal,hi,si,tr=gdn_layer_oracle(raw,cfg,x.cpu(),s.history.cpu(),s.recurrent.cpu())
  p.run(x,out=o,history=s.history,state=s.recurrent,history_out=h,state_out=st,workspace=w)
  alltr=dict(normalized=w.normalized,qkv=w.qkv,z=w.z,a=w.a,b=w.b,convolved=w.mixer.convolved,log_decay=w.mixer.log_decay,beta=w.mixer.beta,recurrent=w.mixer.recurrent,mixed=w.mixed,projected=w.projected,residual=w.residual,mlp_activated=w.mlp.gate,mlp_up=w.mlp.up,mlp_down=w.mlp.hidden,output=o,history=h,state=st)
  tr.update(output=ideal,history=hi,state=si)
  case=dict(name=name,stages=[])
  for k,a in alltr.items():
   b=tr[k].double();a=a.cpu().double();err=(a-b).abs()
   case['stages'].append(dict(stage=k,max_abs_error=err.max().item(),mismatches=int((err>0).sum()), first_indices=(err>0).nonzero()[:4].tolist(), first_values=[[a[tuple(i)].item(),b[tuple(i)].item()] for i in (err>0).nonzero()[:4].tolist()]))
  pos=0;chunks=[];trchunks={k:[] for k in alltr if k not in ('state','history')};keep=[]
  for n in (1,2,6):
   cp=r.prepare(tokens=n);cw=cp.empty_workspace();co=torch.empty_like(x[pos:pos+n]);
   cp.run(x[pos:pos+n],out=co,history=s.history,state=s.recurrent,history_out=s.history,state_out=s.recurrent,workspace=cw)
   ctr=dict(normalized=cw.normalized,qkv=cw.qkv,z=cw.z,a=cw.a,b=cw.b,convolved=cw.mixer.convolved,log_decay=cw.mixer.log_decay,beta=cw.mixer.beta,recurrent=cw.mixer.recurrent,mixed=cw.mixed,projected=cw.projected,residual=cw.residual,mlp_activated=cw.mlp.gate,mlp_up=cw.mlp.up,mlp_down=cw.mlp.hidden,output=co)
   for k,v in ctr.items():trchunks[k].append(v)
   keep.append((cp,cw));chunks.append(co);pos+=n
  case['chunk_stages']=[]
  for k,vs in trchunks.items():
   a,b=torch.cat(vs).cpu().double(),alltr[k].cpu().double();err=(a-b).abs()
   case['chunk_stages'].append(dict(stage=k,max_abs_error=err.max().item(),mismatches=int((err>0).sum()),first_indices=(err>0).nonzero()[:4].tolist()))
  case['state_max_chunk_error']=(s.recurrent.cpu().double()-st.cpu().double()).abs().max().item()
  report.append(case)
(root/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
