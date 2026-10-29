"""Explicit original Flash-Next: current weight-transfer vs fixed-hot hybrid.

CPU packed weights are additional bounded RAM (up to48GiB), NOT a free speedup.
Cross-device FP16 boundaries, FP32 CPU BLAS; no native quantized CPU kernel or
bitwise-equivalence claim. Correctness stage must finish before score stage.
"""
import argparse,gc,hashlib,json,pathlib,time
from dataclasses import asdict
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from benchmark_flash_next_full import checked_payloads,process_memory
from flash_next_hybrid_metrics import GATES,difference,state_to_cpu,compare_state
from venturi.checkpoint import inspect_checkpoint
from venturi.generation import LocalTokenizer
from venturi.flash_next_generation import generate_flash_next_text
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_hybrid_moe import bind_fixed_expert_policy,RouteCounter
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_request import FlashNextRequest

PROMPT='<|im_start|>user\n用一句话介绍你自己。<|im_end|>\n<|im_start|>assistant\n'
MODES=('current-gpu-lru','fixed-hot-gpu-cold','fixed-hot-cpu-cold')


def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def memory():
 result=process_memory()
 result['current_rss_bytes']=next(int(line.split()[1])*1024 for line in pathlib.Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['checkpoint','library','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--stage',choices=['correctness','benchmark','profile'],required=True)
 p.add_argument('--qualification');p.add_argument('--workers',type=int,choices=[1,4,8],default=8)
 p.add_argument('--cpu-cache-mib-per-layer',type=int,choices=[0,128,256,512,1024],default=1024)
 p.add_argument('--cpu-compute',choices=['staged-fp32','native-half'],default='staged-fp32')
 p.add_argument('--profile-mode',choices=MODES,default='fixed-hot-cpu-cold')
 a=p.parse_args();directory,out=pathlib.Path(a.checkpoint),pathlib.Path(a.output);out.mkdir(exist_ok=False)
 torch.set_num_threads(1);device=torch.device('cuda:0');props=torch.cuda.get_device_properties(device)
 assert (props.major,props.minor)==(7,0);torch.cuda.set_per_process_memory_fraction(.85)
 proof,manifest=checked_payloads(directory);cp=inspect_checkpoint(directory);n=len(cp.config['text_config']['layer_types']);assert n==48
 token=LocalTokenizer(directory/'tokenizer.json',vocab_size=cp.config['text_config']['vocab_size'])
 start=time.perf_counter();loaded=load_weight_decoded_fp8_text(cp,max_core_bytes=12*2**30,max_expert_bytes=64*2**20,max_sequence=256,attention_backend='sdpa',execution_device='cuda:0',execution_dtype=torch.float16,max_execution_core_bytes=12*2**30);torch.cuda.synchronize()
 core=loaded.core;report=dict(scope=__doc__,stage=a.stage,manifest_sha256=proof['manifest_sha256'],revision=manifest['hf_revision'],dtype='FP16 execution, FP8 decoded weights; CPU compute mode explicit',cpu_compute_mode=a.cpu_compute,gates=GATES,prompt=PROMPT,prompt_ids=token.encode(PROMPT),prefill_chunk=16,context_reservation=256,concurrency=1,output_tokens=32,decode_transitions=31,workers=a.workers,intraop_threads=1,cpu_cache_mib_per_layer=a.cpu_cache_mib_per_layer,cpu_cache_total_budget=a.cpu_cache_mib_per_layer*2**20*n,loading_seconds=time.perf_counter()-start,records=[],native_full_model=False,throughput_scored=False)
 binding=None;hot=None
 def save():write(out/'report.json',report)
 def close():
  nonlocal binding
  if binding is not None:binding.close();binding=None
  for layer in core.layers:
   layer.mlp.expert_dispatch=None
   if hasattr(layer.mlp.expert_store,'clear'):layer.mlp.expert_store.clear()
  gc.collect()
 def activate(mode):
  nonlocal binding
  close();start=time.perf_counter()
  if mode=='current-gpu-lru':
   plan=bind_fp8_expert_policy(core,cp,backend='cuda-byte-decode',retained_bytes_per_layer=(256*2**20,)*n,library=a.library)
  else:
   binding=bind_fixed_expert_policy(core,cp,mode=mode,hot_ids_per_layer=hot,hot_bytes_per_layer=256*2**20,workers=a.workers,library=a.library,cpu_cache_bytes_per_layer=a.cpu_cache_mib_per_layer*2**20,cpu_compute_mode=a.cpu_compute)
   plan=dict(mode=mode,hot_ids_per_layer=hot,hot_gpu_budget=256*2**20*n,cpu_cache_budget=report['cpu_cache_total_budget'] if mode=='fixed-hot-cpu-cold' else 0)
  torch.cuda.synchronize();return dict(plan=plan,setup_seconds=time.perf_counter()-start)
 def stats():
  if binding is not None:return binding.stats()
  return dict(mode='current-gpu-lru',layers=[dict(layer=i,cache=layer.mlp.expert_store.stats) for i,layer in enumerate(core.layers)])
 save()
 try:
  with PreparedNGramTable(next(iter(loaded.tables.values()))) as reader:
   if a.stage=='correctness':
    activate(MODES[0]);counters=[]
    for layer in core.layers:
     counter=RouteCounter(layer.mlp.expert_store);layer.mlp.expert_dispatch=counter;counters.append(counter)
    calibration=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=4,eos_token_ids=(),prefill_chunk_size=16)
    count=256*2**20//(3*2560*640*2);assert count==27
    hot=[list(counter.top(count)) for counter in counters]
    counts=[counter.counts for counter in counters]
    for layer in core.layers:layer.mlp.expert_dispatch=None
    del counters,counter
    hot_plan=dict(manifest_sha256=proof['manifest_sha256'],workers=a.workers,cpu_cache_mib_per_layer=a.cpu_cache_mib_per_layer,cpu_compute_mode=a.cpu_compute,hot=hot,counts=counts,calibration=asdict(calibration),scope='Same-prompt unscored13-token prefill+3 decode transitions ONLY; freeze before scoring future32 tokens; not unseen-workload evidence')
    write(out/'hot-plan.json',hot_plan);report['hot_plan_sha256']=hashlib.sha256((out/'hot-plan.json').read_bytes()).hexdigest();report['hot_plan']=hot_plan;save()
    baseline=[];checks=[];route_traces={};generation_match=True
    for mode in MODES:
     setup=activate(mode);request=FlashNextRequest(core,reader)
     traces=[[] for _ in core.layers];handles=[]
     for i,layer in enumerate(core.layers):
      def capture(module,args,output,index=i):traces[index].append(output[2].detach().cpu().tolist())
      handles.append(layer.mlp.gate.register_forward_hook(capture))
     with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
      for step,ids in enumerate(([248045],[846],[198])):
       proposal=request.preview(ids);request.commit(proposal);logits=proposal.logits.cpu();state=state_to_cpu(request.state)
       if mode==MODES[0]:baseline.append((logits,state));continue
       expected,expected_state=baseline[step];delta=difference(logits,expected,kind='logits');states=compare_state(state,expected_state)
       passed=delta['passed'] and all(x['passed'] for x in states) and logits.argmax(-1).tolist()==expected.argmax(-1).tolist()
       exact=delta['exact'] and all(x['exact'] for x in states)
       if mode==MODES[1]:passed=passed and exact
       checks.append(dict(mode=mode,step=step,tokens=ids,passed=passed,exact=exact,logits=delta,states=states));report['checks']=checks;save()
       if not passed:
        torch.save(dict(actual_logits=logits,expected_logits=expected,actual_state=state,expected_state=expected_state),out/f'counterexample-{mode}-step{step}.pth')
        report['qualification_failed']=True;save()
     for handle in handles:handle.remove()
     route_traces[mode]=traces;report['route_traces']=route_traces;save()
     del request,proposal,state,logits
     if mode in [MODES[0],MODES[2]]:
      result=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
      report.setdefault('generation_checks',[]).append(dict(mode=mode,setup=setup,result=asdict(result),expert_stats=stats()));save()
      if mode==MODES[0]:expected_ids=result.generated_ids
      elif result.generated_ids!=expected_ids:
       generation_match=False;report['qualification_failed']=True;save()
    report['all_six_logit_state_checks_passed']=all(x['passed'] for x in checks);report['generated_ids_exact']=generation_match;report['expected_generated_ids']=list(expected_ids)
    report['completed']=report['all_six_logit_state_checks_passed'] and generation_match;save()
    if not report['completed']:raise AssertionError('Strict qualification failed; diagnostic generation retained, performance remains forbidden')
    return
   qualification=pathlib.Path(a.qualification);q=json.loads((qualification/'report.json').read_text());plan=json.loads((qualification/'hot-plan.json').read_text())
   if not q.get('completed') or not q.get('generated_ids_exact') or q['manifest_sha256']!=proof['manifest_sha256'] or hashlib.sha256((qualification/'hot-plan.json').read_bytes()).hexdigest()!=q['hot_plan_sha256']:
    raise ValueError('Completed matching correctness/greedy/hot-set proof required before performance')
   if plan['workers']!=a.workers or plan['cpu_cache_mib_per_layer']!=a.cpu_cache_mib_per_layer or plan['cpu_compute_mode']!=a.cpu_compute:raise ValueError('Performance parameters differ from qualified hybrid')
   hot=plan['hot'];report['qualification_report_sha256']=hashlib.sha256((qualification/'report.json').read_bytes()).hexdigest();report['hot_plan_sha256']=q['hot_plan_sha256'];report['expected_generated_ids']=q['expected_generated_ids'];save()
   if a.stage=='profile':
    setup=activate(a.profile_mode)
    if binding is not None and binding.pool is not None:
     binding.pool.trace=True
     for dispatch in binding.dispatches:dispatch.trace=True
    # Independent warmup outside profiler NVTX range, same shape/count as scores.
    generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=4,eos_token_ids=(),prefill_chunk_size=16)
    with torch.cuda.nvtx.range('venturi.hybrid.profile'):
     result=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=4,eos_token_ids=(),prefill_chunk_size=16)
    report.update(completed=True,profile_mode=a.profile_mode,profile_setup=setup,profile_result=asdict(result),expert_stats=stats());save();return
   orders=[MODES,tuple(reversed(MODES)),(MODES[1],MODES[0],MODES[2])]
   for repeat,order in enumerate(orders):
    for mode in order:
     setup=activate(mode);warmup=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=4,eos_token_ids=(),prefill_chunk_size=16)
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=memory();epoch=time.time()
     result=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
     finished=time.time();after=memory()
     if list(result.generated_ids)!=q['expected_generated_ids']:raise AssertionError('Scored variant greedy IDs changed; preserve partial data, not a valid speedup')
     record=dict(mode=mode,repeat=repeat,setup=setup,warmup_output_tokens=len(warmup.generated_ids),result=asdict(result),decode_tps=result.decode_tokens/result.decode_seconds,request_tps=len(result.generated_ids)/result.total_seconds,scored_started_epoch=epoch,scored_finished_epoch=finished,memory_before=before,memory_after=after,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),expert_stats=stats())
     report['records'].append(record);save();print(json.dumps({k:record[k] for k in ['mode','repeat','decode_tps','request_tps']}),flush=True)
   report['summary']=[dict(mode=mode,repeats=3,decode_tps=sum(x['result']['decode_tokens'] for x in report['records'] if x['mode']==mode)/sum(x['result']['decode_seconds'] for x in report['records'] if x['mode']==mode),request_tps=sum(len(x['result']['generated_ids']) for x in report['records'] if x['mode']==mode)/sum(x['result']['total_seconds'] for x in report['records'] if x['mode']==mode)) for mode in MODES]
   report.update(completed=True,throughput_scored=True,all_generated_ids_exact=True);save()
 finally:close()


if __name__=='__main__':main()
