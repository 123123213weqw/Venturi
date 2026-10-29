"""Actual-weight bounded MoE CPU/GPU gate and small-operator comparison; NOT TPS."""
import argparse,json,pathlib,time,resource
import torch
from flash_next_hybrid_metrics import difference,GATES
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_moe import MoEGeometry,routed_experts
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_cuda_fp8 import CUDABlockFP8ExpertStore
from venturi.ops.flash_next_fp8 import NativeBlockFP8Decoder
from venturi.models.flash_next_hybrid_moe import ColdExpertPool,FixedHotBank,HybridExpertDispatch,CPUDecodedExpertCache


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--library',required=True);p.add_argument('--output',required=True);p.add_argument('--cpu-cache-mib',type=int,choices=[0,256],default=0);p.add_argument('--cpu-compute',choices=['staged-fp32','native-half'],default='staged-fp32');a=p.parse_args();out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
 torch.set_num_threads(1);torch.manual_seed(2291);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(device).total_memory)
 cp=inspect_checkpoint(a.checkpoint);g=MoEGeometry.from_dict(dict(cp.config));assert (g.hidden_size,g.intermediate_size,g.top_k)==(2560,640,10)
 decoder=NativeBlockFP8Decoder(a.library);records=[];weight_checks=[];measurements=[]
 report=dict(scope=__doc__,gates=GATES,real_model_tps=False,activation_quantization=False,cpu_cache_mib=a.cpu_cache_mib,cpu_compute_mode=a.cpu_compute,records=records,weight_checks=weight_checks,measurements=measurements)
 def save():(out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
 for layer in [0,3]:
  prefix=f'model.language_model.layers.{layer}.mlp.experts';cpu=BlockFP8ExpertStore(cp,g,prefix,decode_strategy='vectorized');gpu=CUDABlockFP8ExpertStore(cp,g,prefix,decoder=decoder)
  selected=[0,1,7,13,31,63,127,255,383,511];hot=selected[::2];bank=FixedHotBank(gpu,hot,device=device,dtype=torch.float16,max_bytes=64*2**20)
  for e in selected:
   ca=cpu.read(e,device='cpu',dtype=torch.float16);ga=gpu.read(e,device=device,dtype=torch.float16);exact=all(torch.equal(x,y.cpu()) for x,y in zip(ca,ga));weight_checks.append(dict(layer=layer,expert=e,decoded_weights_exact=exact));save();assert exact
  prepared_cpu=CPUDecodedExpertCache(cpu,max_bytes=a.cpu_cache_mib*2**20,compute_mode=a.cpu_compute)
  for tokens in [1,4,13]:
   x=(torch.randn(tokens,g.hidden_size,device=device)*.1).half();ids=torch.tensor([selected[i:]+selected[:i] for i in [i%10 for i in range(tokens)]],device=device);scores=torch.softmax(torch.randn(tokens,10,device=device),-1).half()
   with torch.inference_mode():expected=routed_experts(x,ids,scores,bank)
   for workers in [1,4,8]:
    pool=ColdExpertPool(workers);dispatch=HybridExpertDispatch(bank,prepared_cpu,pool)
    try:
     actual=dispatch(x,ids,scores);delta=difference(actual,expected,kind='component');records.append(dict(layer=layer,tokens=tokens,workers=workers,numerical=delta,dispatch=dict(dispatch.stats)));save()
     if not delta['passed']:
      torch.save(dict(actual=actual.cpu(),expected=expected.cpu(),x=x.cpu(),ids=ids.cpu(),scores=scores.cpu()),out/f'counterexample-layer{layer}-n{tokens}-w{workers}.pth');raise AssertionError('Actual-weight hybrid MoE numerical gate failed')
     for repeat in range(3):
      torch.cuda.synchronize();wall,ct=time.perf_counter(),time.process_time();actual=dispatch(x,ids,scores);torch.cuda.synchronize();measurements.append(dict(layer=layer,tokens=tokens,workers=workers,repeat=repeat,mode='hybrid',seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-ct))
    finally:pool.close()
   for repeat in range(3):
    torch.cuda.synchronize();wall,ct=time.perf_counter(),time.process_time()
    with torch.inference_mode():routed_experts(x,ids,scores,bank)
    torch.cuda.synchronize();measurements.append(dict(layer=layer,tokens=tokens,repeat=repeat,mode='matched-gpu-cold',seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-ct))
   report['last_cpu_cache_stats']=prepared_cpu.stats;save()
  bank.clear()
 report.update(completed=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated());save();print(json.dumps(dict(completed=True,cases=len(records),weight_checks=len(weight_checks),measurements=len(measurements))))


if __name__=='__main__':main()
