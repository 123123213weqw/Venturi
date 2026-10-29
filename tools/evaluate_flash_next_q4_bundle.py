"""Complete Q4 bundle: real reload gate, fresh-process/cohort TPS and held-out screen.

Reference PyTorch execution on one V100; no BF16/official W8A8/native-engine claim.
No inference quantization. Fixed-token speed probes are NOT answer-quality scores.
"""
import argparse,gc,hashlib,json,math,os,time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import save_file,load_file
from benchmark_flash_next_full import checked_payloads,process_memory
from flash_next_hybrid_metrics import difference,state_to_cpu,compare_state
from venturi.checkpoint import inspect_checkpoint
from venturi.flash_next_generation import generate_flash_next_text,generate_flash_next_tokens,resolve_flash_next_eos
from venturi.generation import LocalTokenizer
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text,load_weight_decoded_q4_text
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_moe import MoEGeometry
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_q4 import LazyQ4ExpertArchive,PackedQ4Dispatch
from venturi.models.flash_next_q4_bundle import bundle_manifest
from venturi.models.flash_next_request import FlashNextRequest
from venturi.ops.flash_next_q4 import NativeQ4


def write(path,value):
 t=path.with_suffix('.json.part');t.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');t.replace(path)


def chat(text,*,thinking):
 return '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n'+('<think>\n' if thinking else '<think>\n\n</think>\n\n')


def grade(value,text):
 if value['grader']=='json':
  try:
   actual=json.loads(text);expected=value['answer']
   return type(actual) is dict and actual.keys()==expected.keys() and all(type(actual[k]) is type(v) and actual[k]==v for k,v in expected.items())
  except (ValueError,TypeError):return False
 return text.strip()==value['answer']


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['checkpoint','bundle','fp8-library','q4-library','suite','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--stage',choices=['correctness','cohort','startup-repeat','quality','profile'],required=True)
 p.add_argument('--backend',choices=['fp8','q4'],default='q4');p.add_argument('--round',type=int,default=0)
 p.add_argument('--spawn-epoch',type=float,default=0);p.add_argument('--reference');p.add_argument('--old-archive');p.add_argument('--qualification')
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);entered=time.time()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=torch.device('cuda:0')
 assert torch.cuda.get_device_capability(device)==(7,0);torch.cuda.set_per_process_memory_fraction(.85)
 suite=json.loads(Path(a.suite).read_text());bundle=bundle_manifest(a.bundle)
 if a.backend=='fp8' or a.stage=='correctness':proof,source=checked_payloads(Path(a.checkpoint))
 else:proof,source=dict(manifest_sha256=bundle['source_identity']['manifest_sha256']),dict(hf_revision=bundle['source_identity']['revision'])
 if bundle['source_identity']['manifest_sha256']!=proof['manifest_sha256']:raise ValueError('Source/bundle provenance mismatch')
 opts=dict(max_core_bytes=12*2**30,max_expert_bytes=64*2**20,max_sequence=512,attention_backend='sdpa',execution_device='cuda:0',execution_dtype=torch.float16,max_execution_core_bytes=12*2**30)
 report=dict(scope=__doc__,stage=a.stage,backend=a.backend,round=a.round,source_revision=source['hf_revision'],source_manifest_sha256=proof['manifest_sha256'],bundle_manifest_sha256=hashlib.sha256((Path(a.bundle)/'bundle.json').read_bytes()).hexdigest(),suite_sha256=hashlib.sha256(Path(a.suite).read_bytes()).hexdigest(),entered_main_epoch=entered,spawn_epoch=a.spawn_epoch,context_reservation=512,prefill_chunk=16,concurrency=1,weight_cache_budget_per_layer=256*2**20,source='FP8 -> FP16 -> RTN Q4; staticBF16/PLEFP8 unchanged',native_full_model=False,completed=False,records=[])
 def save():write(out/'report.json',report)
 save()
 if a.stage not in ('correctness','quality'):
  q=json.loads(Path(a.qualification).read_text());assert q['completed'] and q['bundle_manifest_sha256']==report['bundle_manifest_sha256'] and all(x['exact'] for x in q['checks'])
 start,ct=time.perf_counter(),time.process_time();cp=inspect_checkpoint(a.bundle if a.backend=='q4' else a.checkpoint)
 loaded=load_weight_decoded_q4_text(cp,**opts) if a.backend=='q4' else load_weight_decoded_fp8_text(cp,**opts)
 core=loaded.core;dispatches=[]
 if a.backend=='fp8':bind_fp8_expert_policy(core,cp,backend='cuda-byte-decode',retained_bytes_per_layer=(256*2**20,)*len(core.layers),library=a.fp8_library)
 else:
  native=NativeQ4(a.q4_library)
  for layer in core.layers:
   d=PackedQ4Dispatch(layer.mlp.expert_store,native,device=device,max_resident_bytes=256*2**20);dispatches.append(d);layer.mlp.expert_dispatch=d.decoded_reference
 token=LocalTokenizer(cp.directory/'tokenizer.json',vocab_size=cp.config['text_config']['vocab_size']);eos,eos_source=resolve_flash_next_eos(cp)
 torch.cuda.synchronize();report.update(model_load_seconds=time.perf_counter()-start,model_load_cpu_seconds=time.process_time()-ct,model_ready_epoch=time.time(),loading_plan=dict(loaded.report),tokenizer_sha256=token.sha256,eos=eos,eos_source=eos_source)
 def stats():return [d.stats for d in dispatches] if dispatches else [layer.mlp.expert_store.stats for layer in core.layers]
 def run(text,n,*,thinking,stop):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=process_memory();epoch=time.time()
  tok_start=time.perf_counter();ids=token.encode(chat(text,thinking=thinking));tokenization_seconds=time.perf_counter()-tok_start;generation_epoch=time.time()
  result=generate_flash_next_tokens(core,reader,ids,max_new_tokens=n,eos_token_ids=eos if stop else (),prefill_chunk_size=16)
  finish=time.time();after=process_memory();value=asdict(result);value['text']=token.decode(result.generated_ids)
  generated=list(result.generated_ids);clean=token.decode(generated[:-1] if generated and generated[-1] in eos else generated)
  row=dict(result=value,clean_text=clean,start_epoch=epoch,finish_epoch=finish,tokenization_seconds=tokenization_seconds,request_ttft_seconds=generation_epoch-epoch+result.time_to_first_token_seconds,first_token_epoch=generation_epoch+result.time_to_first_token_seconds,process_start_to_first_token_seconds=generation_epoch+result.time_to_first_token_seconds-a.spawn_epoch if a.spawn_epoch else None,decode_tps=result.decode_tokens/result.decode_seconds if result.decode_tokens else None,request_tps=len(generated)/result.total_seconds,memory_before=before,memory_after=after,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),expert_stats=stats(),contains_eos=any(t in eos for t in generated),fixed_token_ignore_eos=not stop)
  return row
 try:
  prep=time.perf_counter()
  with PreparedNGramTable(next(iter(loaded.tables.values()))) as reader:
   report['reader_preparation_seconds']=time.perf_counter()-prep;save()
   if a.stage=='correctness':
    # Independently load ORIGINAL static core; original Q4 bank is read-only.
    original=load_weight_decoded_fp8_text(inspect_checkpoint(a.checkpoint),**opts)
    g=MoEGeometry.from_dict(dict(cp.config))
    class ReadOnlySample(LazyQ4ExpertArchive):
     def packed(self,e):
      if str(e) not in self.manifest['experts']:raise ValueError('Old sample missing; never mutate old evidence/archive')
      return super().packed(e)
    for i,layer in enumerate(original.core.layers):
     layer.mlp.expert_store=ReadOnlySample(SimpleNamespace(geometry=g),Path(a.old_archive)/f'layer-{i:02d}',source_identity=dict(bundle['source_identity'],layer=i))
    # Zero retained expert cache while holding TWO9.2GiB cores for this gate.
    for d in dispatches:d.max_resident_bytes=0
    checks=[]
    with PreparedNGramTable(next(iter(original.tables.values()))) as oldreader,torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
     old,new=FlashNextRequest(original.core,oldreader),FlashNextRequest(core,reader)
     for step,ids in enumerate(([248045],[846],[198])):
      x,y=old.preview(ids),new.preview(ids);old.commit(x);new.commit(y)
      error=difference(y.logits.cpu(),x.logits.cpu(),kind='logits');states=compare_state(state_to_cpu(new.state),state_to_cpu(old.state));exact=error['exact'] and all(v['exact'] for v in states)
      checks.append(dict(step=step,exact=exact,logits=error,states=states));report['checks']=checks;save()
      if not exact:
       torch.save(dict(expected=x.logits.cpu(),actual=y.logits.cpu(),expected_state=state_to_cpu(old.state),actual_state=state_to_cpu(new.state)),out/f'counterexample-export-step{step}.pth');raise AssertionError('Export/source complete text states must be bit-exact')
    report['completed']=True;save();return
   if a.stage in ('cohort','startup-repeat'):
    cases=suite['performance'];offset=a.round%len(cases);cases=cases[offset:]+cases[:offset]
    if a.stage=='startup-repeat':cases=cases[:1]
    for item in cases:
     row=run(item['text'],32,thinking=True,stop=False);row.update(id=item['id'],prompt=item['text']);report['records'].append(row);save()
   elif a.stage=='quality':
    reference={} if a.backend=='fp8' else load_file(a.reference,device='cpu')
    nll=kl_sum=0.;targets=agree=0;teacher=[]
    for passage in suite['passages']:
     ids=token.encode(passage['text']);assert len(ids)>=65
     ids=ids[:65];request=FlashNextRequest(core,reader);parts=[]
     with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
      for start in range(0,64,16):
       prop=request.preview(ids[start:start+16]);request.commit(prop);logits=prop.logits[0].float();gold=torch.tensor(ids[start+1:start+17],device=device)
       losses=F.cross_entropy(logits,gold,reduction='none');nll+=float(losses.sum());targets+=len(gold);record=dict(passage=passage['id'],offset=start,target_ids=gold.cpu().tolist(),nll=losses.cpu().tolist())
       if a.backend=='fp8':parts.append(prop.logits[0].cpu())
       else:
        ref=reference[passage['id']][start:start+16].to(device).float();lp,lq=F.log_softmax(ref,-1),F.log_softmax(logits,-1);kl=(lp.exp()*(lp-lq)).sum(-1);kl_sum+=float(kl.sum());same=(ref.argmax(-1)==logits.argmax(-1));agree+=int(same.sum());record.update(kl=kl.cpu().tolist(),top1_same=same.cpu().tolist())
       teacher.append(record)
     if a.backend=='fp8':reference[passage['id']]=torch.cat(parts)
     report['teacher_records']=teacher;save();del request
    if a.backend=='fp8':save_file(reference,str(out/'reference.safetensors'))
    report['teacher_summary']=dict(targets=targets,mean_nll=nll/targets,perplexity=math.exp(nll/targets),mean_kl=kl_sum/targets if a.backend=='q4' else None,top1_agreement=agree/targets if a.backend=='q4' else None,scope='256 manually authored held-out plain-text target tokens, not a public benchmark')
    for item in suite['closed']+suite['open']:
     closed='grader' in item;row=run(item['text'],96 if closed else 128,thinking=False,stop=True);row.update(id=item['id'],prompt=item['text'],closed=closed)
     if closed:row['passed']=row['result']['finish_reason']=='eos' and grade(item,row['clean_text']);row['expected']=item['answer']
     report['records'].append(row);save()
    closed=[x for x in report['records'] if x['closed']];report['closed_score']=sum(x['passed'] for x in closed)/len(closed)
   elif a.stage=='profile':
    # Explicit separate profile, unrelated warmup; not any scored interval.
    run('你好。',4,thinking=True,stop=False)
    with torch.cuda.nvtx.range('venturi.q4.export-heldout-profile'):
     row=run(suite['performance'][0]['text'],4,thinking=True,stop=False)
    report['profile_record']=row
   report.update(completed=True,inference_converted_experts=sum(layer.mlp.expert_store.converted for layer in core.layers) if a.backend=='q4' else 0)
   assert report['inference_converted_experts']==0
   save()
 finally:
  report.update(process_cpu_seconds_at_exit=time.process_time(),memory_at_exit=process_memory());save()
  for d in dispatches:d.clear()

if __name__=='__main__':main()
