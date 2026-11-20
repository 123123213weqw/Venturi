"""Qwen3.8 fixed/adaptive fresh-cohort MTP, gated on exact target IDs/private states."""
import argparse,json,hashlib,threading,subprocess
from pathlib import Path
from time import perf_counter,process_time,time
import torch
from venturi.models import NativeQwenModel,NativeQwenMTPModel,QwenQ8Weights,QwenMTPWeights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend
from venturi.adaptive_mtp import AdaptiveMTPPolicy
from benchmark_model import signature,write,PROFILE
p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--draft',required=True);p.add_argument('--library',required=True);p.add_argument('--prompts',required=True);p.add_argument('--output',required=True);p.add_argument('--reference',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(4)
ref=json.loads(Path(a.reference).read_text());assert ref['completed'] and ref['exact_reference_bytes_equal']
prompts=json.loads(Path(a.prompts).read_text())['prompt_token_ids'];rows=[];gpu=[];stop=threading.Event()
def sample():
 while not stop.is_set():
  r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
  gpu.append(dict(unix=time(),rc=r.returncode,csv=r.stdout));stop.wait(.5)
target=model=adaptive=None;thread=threading.Thread(target=sample);thread.start()
try:
 target=NativeQwenModel.from_q8(QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact)),CudaBackend(a.library),weight_mode='resident',dtype=torch.float16,max_tokens=32,
  reserve_bytes=256*2**20,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',projection_profile=PROFILE)
 model=NativeQwenMTPModel.from_q8(target,QwenMTPWeights.from_ninfer(NInferArtifact(a.draft)),draft_projection_profile='q8-residual')
 policy=AdaptiveMTPPolicy();adaptive=model.create_adaptive_batch([256]*8,policy=policy)
 start=perf_counter();adaptive.prepare();write(out/'config.json',dict(args=vars(a),preparation_seconds=perf_counter()-start,target_id=target._weights.artifact.artifact_id,draft_id=model._mtp_weights.artifact.artifact_id,
   library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),banks={d:b.info() for d,b in adaptive._banks.items()},scope=__doc__))
 for count in [1,8]:
  depths=[0,1,2] if count==1 else [0,1];inputs=prompts[:count];states=None
  expected=[ref['comparison_payload']['scalar'][0]['tokens']] if count==1 else ref['comparison_payload']['cohort']['tokens']
  def run(phase,depth=None,learn=False):
   global states
   wall,cpu,started=perf_counter(),process_time(),time()
   ids=adaptive.generate(inputs,max_new_tokens=32,eos_token_ids=[],strategy=depth,learn=learn)
   elapsed,used=perf_counter()-wall,process_time()-cpu;info=adaptive.info();selected=info['selected_depth'];bank=adaptive._banks[selected]
   assert ids==expected,'MTP changed exact ordinary target trajectory'
   # State hashing is outside the scored whole-call wall/CPU interval.
   state=[signature(bank.snapshot())] if selected==2 else [signature(bank.snapshot(s)) for s in range(count)]
   if states is None:states=state
   assert state==states,'MTP committed private target state differs'
   row=dict(count=count,phase=phase,requested_depth=depth,selected_depth=selected,started_unix=started,wall_seconds=elapsed,process_cpu_seconds=used,process_cpu_cores=used/elapsed,
      aggregate_tps=sum(map(len,ids))/elapsed,emitted_tokens=sum(map(len,ids)),outputs=ids,exact_target_state=True,info=info)
   rows.append(row);write(out/'rows.json',rows)
  for d in depths:run('correctness-warmup',d)
  for _ in depths:run('cold-exploration',learn=True)
  for _ in range(2):
   for d in depths:run('calibration',d,True)
  for repeat in range(3):
   order=depths+[None]
   if repeat%2:order.reverse()
   for d in order:run('timed-auto' if d is None else 'timed-fixed',d,d is None)
 write(out/'completed.json',dict(completed=True,counts=[1,8],repeats=3,policy=policy.info(),comparison='Whole fresh cohort including policy/reset/native draft+verification+metadata, no HTTP; no post-first TPS claim',state_scope='Exact target state and greedy IDs for these 8 prompts only; draft quality is not assumed from acceptance',unix=time()))
finally:
 stop.set();thread.join(timeout=10);write(out/'gpu-telemetry.json',gpu)
 if adaptive is not None:adaptive.close()
 if model is not None:model.close()
 if target is not None:target.close()
