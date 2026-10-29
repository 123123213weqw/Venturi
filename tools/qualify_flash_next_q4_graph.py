"""Separate correctness ONLY: actual packed experts then full48-layer Q4 states.

No component timings are model TPS. Explicit prebuilt SM70 decoder; no compile.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from flash_next_hybrid_metrics import difference,state_to_cpu,compare_state
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_q4_bundle import bundle_manifest,complete_stores
from venturi.models.flash_next_loader import load_weight_decoded_q4_text
from venturi.models.flash_next_q4 import PackedQ4Dispatch
from venturi.models.flash_next_q4_graph import CoalescedQ4,CoalescedQ4Dispatch,RoutedQ4Graph
from venturi.models.flash_next_moe import MoEGeometry
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_request import FlashNextRequest
from venturi.ops.flash_next_q4 import NativeQ4
from venturi.generation import LocalTokenizer
from evaluate_flash_next_q4_bundle import chat


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['bundle','library','output']:p.add_argument('--'+n,required=True)
 p.add_argument('--stage',choices=['component','full'],required=True);p.add_argument('--mode',choices=['coalesced','graph'],required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=torch.device('cuda:0');assert torch.cuda.get_device_capability(device)==(7,0)
 torch.cuda.set_per_process_memory_fraction(.06 if a.stage=='component' else .85)
 cp=inspect_checkpoint(a.bundle);m=bundle_manifest(a.bundle);g=MoEGeometry.from_dict(dict(cp.config));native=CoalescedQ4(a.library);oldnative=NativeQ4(a.library);workspace=None
 report=dict(stage=a.stage,mode=a.mode,completed=False,checks=[],bundle_manifest_sha256=hashlib.sha256((Path(a.bundle)/'bundle.json').read_bytes()).hexdigest(),library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),scope=__doc__,started_epoch=time.time())
 def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 def check(x,y,*,label,oldstate=None,newstate=None,extra=None):
  d=difference(y.cpu(),x.cpu(),kind='logits' if a.stage=='full' else 'component');states=[] if oldstate is None else compare_state(state_to_cpu(newstate),state_to_cpu(oldstate));exact=d.get('exact',False) and all(v['exact'] for v in states);row=dict(label=label,exact=exact,difference=d,states=states);report['checks'].append(row);save()
  if not exact:
   torch.save(dict(expected=x.cpu(),actual=y.cpu(),extra=extra,expected_state=state_to_cpu(oldstate) if oldstate else None,actual_state=state_to_cpu(newstate) if newstate else None),out/('counterexample-'+label+'.pth'));raise AssertionError('Strict canonical Q4 equality failed; no TPS scoring for this arm')
 try:
  workspace=RoutedQ4Graph(native,g,device=device) if a.mode=='graph' else None
  if a.stage=='component':
   stores=complete_stores(cp,g,m['layers']);gen=torch.Generator(device=device).manual_seed(2731)
   with torch.inference_mode():
    for layer in [0,23,47]:
     old=PackedQ4Dispatch(stores[layer],oldnative,device=device,max_resident_bytes=64*2**20);new=CoalescedQ4Dispatch(stores[layer],native,device=device,max_resident_bytes=64*2**20,workspace=workspace)
     try:
      for n in [1,1,4,16,1]:
       x=(torch.randn((n,g.hidden_size),generator=gen,device=device)*.1).half();probs=torch.softmax(torch.randn((n,g.num_experts),generator=gen,device=device),-1);scores,ids=probs.topk(g.top_k,-1);scores=(scores/scores.sum(-1,keepdim=True)).half()
       first=old.decoded_reference(x,ids,scores);second=new.decoded_reference(x,ids,scores);check(first,second,label=f'l{layer}-n{n}-case{len(report["checks"])}',extra=dict(input=x.cpu(),indices=ids.cpu(),scores=scores.cpu()))
      ids[:]=ids[:,0:1];first=old.decoded_reference(x,ids,scores);second=new.decoded_reference(x,ids,scores);check(first,second,label=f'l{layer}-duplicates',extra=dict(input=x.cpu(),indices=ids.cpu(),scores=scores.cpu()))
      report.setdefault('cache_stats',[]).append(dict(layer=layer,old=old.stats,new=new.stats));save()
     finally:old.clear();new.clear()
  else:
   opts=dict(max_core_bytes=12*2**30,max_expert_bytes=64*2**20,max_sequence=512,attention_backend='sdpa',execution_device='cuda:0',execution_dtype=torch.float16,max_execution_core_bytes=12*2**30)
   old=load_weight_decoded_q4_text(cp,**opts);new=load_weight_decoded_q4_text(cp,**opts);dispatch=[]
   for layer in old.core.layers:
    d=PackedQ4Dispatch(layer.mlp.expert_store,oldnative,device=device,max_resident_bytes=0);dispatch.append(d);layer.mlp.expert_dispatch=d.decoded_reference
   for layer in new.core.layers:
    d=CoalescedQ4Dispatch(layer.mlp.expert_store,native,device=device,max_resident_bytes=0,workspace=workspace);dispatch.append(d);layer.mlp.expert_dispatch=d.decoded_reference
   token=LocalTokenizer(cp.directory/'tokenizer.json',vocab_size=cp.config['text_config']['vocab_size'])
   try:
    with PreparedNGramTable(next(iter(old.tables.values()))) as r1,PreparedNGramTable(next(iter(new.tables.values()))) as r2,torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
     q1,q2=FlashNextRequest(old.core,r1),FlashNextRequest(new.core,r2)
     for i,ids in enumerate(([248045],[846],[198])):
      x,y=q1.preview(ids),q2.preview(ids);q1.commit(x);q2.commit(y);check(x.logits,y.logits,label=f'cached-{i}',oldstate=q1.state,newstate=q2.state)
     # Independent held-out prompt, exact prefill then8 identical-history steps.
     q1,q2=FlashNextRequest(old.core,r1),FlashNextRequest(new.core,r2);ids=token.encode(chat('简要解释二分查找的原理。',thinking=True));generated=[]
     for start in range(0,len(ids),16):
      x,y=q1.preview(ids[start:start+16]),q2.preview(ids[start:start+16]);q1.commit(x);q2.commit(y);check(x.logits,y.logits,label=f'prefill-{start}',oldstate=q1.state,newstate=q2.state)
     for i in range(8):
      t=int(x.logits[0,-1].argmax());generated.append(t);x,y=q1.preview([t]),q2.preview([t]);q1.commit(x);q2.commit(y);check(x.logits,y.logits,label=f'teacher-{i}',oldstate=q1.state,newstate=q2.state)
     report['teacher_ids']=generated;report['teacher_text']=token.decode(generated)
   finally:
    for d in dispatch:d.clear()
  report['completed']=True
 finally:
  report['finished_epoch']=time.time();report['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved();report['workspace_calls']=workspace.calls if workspace else 0;save()
  if workspace:workspace.close()

if __name__=='__main__':main()
