"""Full native model static / demand GPU KV / forced RAM spill qualification and fixed-work C1/eight-slot timing.

No HTTP, speculation, sampling, Python token loop or production mutation.
VMM supplies contiguous GPU operands; RAM copies are synchronous pressure fallback.
TTFT/post-first decode and complete-cohort calls have distinct denominators.
"""
import argparse
import ctypes as C
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import threading
from time import perf_counter,process_time,time

import torch
from venturi.models import NativeQwenModel,QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend

PROFILE='q8-residual-column-mlp-head-gdn'


def write(p,j):p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')


def signature(v):
    if torch.is_tensor(v):
        t=v.cpu().contiguous()
        if t.is_floating_point():assert torch.isfinite(t).all().item(),'Nonfinite logits/trace/recurrent/KV state is not a valid correctness pass'
        return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(v,dict):return {k:signature(v[k]) for k in sorted(v) if k in ['position','gdn','kv']}
    if isinstance(v,(list,tuple)):return [signature(x) for x in v]
    return v


def measured(call,output_tokens=None):
    started=time();wall,cpu=perf_counter(),process_time();value=call()
    d=dict(started_unix=started,wall_seconds=perf_counter()-wall,process_cpu_seconds=process_time()-cpu)
    d['process_cpu_cores']=d['process_cpu_seconds']/d['wall_seconds']
    if output_tokens is not None:d.update(output_tokens=output_tokens,tps=output_tokens/d['wall_seconds'])
    return value,d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('artifact');p.add_argument('--library',required=True);p.add_argument('--prompts',required=True)
    p.add_argument('--label',required=True);p.add_argument('--output-directory',required=True)
    p.add_argument('--mode',choices=['qualify','timing','profile'],required=True)
    p.add_argument('--mask',type=int,default=0);p.add_argument('--reference');p.add_argument('--trial',type=int,default=0);p.add_argument('--profile-count',type=int,choices=[1,8],default=8)
    p.add_argument('--counter-start',action='store_true');p.add_argument('--new-tokens',type=int,default=32)
    p.add_argument('--kv-mode',choices=['static','vram-first'],default='static');p.add_argument('--kv-budget-mib',type=int,default=0);p.add_argument('--ram-budget-mib',type=int,default=4096);p.add_argument('--capacity',type=int,default=256)
    a=p.parse_args();out=Path(a.output_directory);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);assert 2<=a.new_tokens<=64 and torch.cuda.device_count()==2
    assert all(torch.cuda.get_device_capability(i)==(7,0) for i in [0,1])
    prompts=json.loads(Path(a.prompts).read_text())['prompt_token_ids'];assert len(prompts)==8 and all(prompts)
    stat=Path(a.artifact).stat();weights=QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    r=dict(scope=__doc__,completed=False,profiled=a.mode=='profile',mode=a.mode,args=vars(a),model_label=a.label,
        artifact_id=weights.artifact.artifact_id,artifact_stat=dict(bytes=stat.st_size,inode=stat.st_ino,mtime_ns=stat.st_mtime_ns),
        config=weights.config,library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        prompt_token_ids=prompts,physical_devices=[0,1],device_uuids=[str(torch.cuda.get_device_properties(i).uuid) for i in [0,1]],
        projection_profile=PROFILE,max_projection_rows=32,capacity_per_request=a.capacity,weight_storage='Q8 group32 / FP16 scale',
        activation_precision='F16 values/KV; two residual-A8 activation planes and FP32 scales/accumulations; FP32 GDN/TP partials',
        mtp=False,eos_disabled_for_fixed_work=True,sampling='greedy',new_tokens=a.new_tokens,trials=[])
    stop,samples=threading.Event(),[]
    def sample():
        while not stop.is_set():
            try:
                q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                samples.append(dict(unix=time(),returncode=q.returncode,lines=q.stdout.splitlines()))
            except Exception as e:samples.append(dict(unix=time(),error=repr(e)))
            stop.wait(.2)
    thread=threading.Thread(target=sample);thread.start()
    model=None;sessions=[];batch=None
    try:
        model,load_timing=measured(lambda:NativeQwenModel.from_q8(weights,CudaBackend(a.library),weight_mode='resident',dtype=torch.float16,
            device='cuda:0',max_tokens=32,reserve_bytes=256*2**20,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',projection_profile=PROFILE,experimental_flags=a.mask,kv_cache=a.kv_mode,kv_gpu_budget_bytes=a.kv_budget_mib*2**20,kv_ram_budget_bytes=a.ram_budget_mib*2**20 if a.kv_mode!='static' else 0))
        r.update(load_timing=load_timing,projection_info=model.projection_info)
        if a.mode=='qualify':
            payload=dict(teacher=[],scalar=[],cohort={});teacher_last=[]
            s=model.create_session(a.capacity);sessions.append(s)
            for prompt in prompts[:2]:
                trace=[];logits=s.forward(prompt,all_logits=True,trace=trace)
                teacher_last.append(logits[-1].clone())
                payload['teacher'].append(dict(logits=signature(logits),trace=[signature(v['hidden']) for v in trace],state=signature(s.snapshot())))
                s.reset();del logits,trace;gc.collect()
                tokens=s.generate(prompt,max_new_tokens=a.new_tokens,eos_token_ids=[])
                payload['scalar'].append(dict(tokens=tokens,state=signature(s.snapshot())))
                s.reset()
            batch=model.create_batch([a.capacity]*8)
            logits=batch.forward(prompts)
            payload['cohort']['teacher_logits']=signature(logits)
            comparisons=[]
            for j,scalar in enumerate(teacher_last):
                delta=(logits[j].double()-scalar.double())
                relative_l2=(delta.norm()/(scalar.double().norm()+1e-12)).item()
                assert relative_l2<0.005,'Unexpected scalar/cohort teacher logit divergence'
                comparisons.append(dict(relative_l2=relative_l2,max_abs=delta.abs().max().item()))
            payload['cohort']['scalar_vs_batch_teacher']=comparisons
            payload['cohort']['teacher_states']=[signature(batch.snapshot(j)) for j in range(8)]
            del logits;batch.reset()
            tokens=batch.generate(prompts,max_new_tokens=a.new_tokens,eos_token_ids=[])
            payload['cohort'].update(tokens=tokens,states=[signature(batch.snapshot(j)) for j in range(8)])
            assert all(payload['scalar'][j]['tokens']==tokens[j] for j in range(2)),'Scalar/cohort greedy trajectories differ'
            r['finite_logits_trace_state_checked']=True
            r['scalar_vs_batch_first_two_trajectories_equal']=True
            r['comparison_payload']=payload
            if a.reference:
                ref=json.loads(Path(a.reference).read_text());assert ref['completed'] and ref['mode']=='qualify'
                for key in ['artifact_id','config','prompt_token_ids','projection_profile','new_tokens']:assert r[key]==ref[key],key
                assert payload==ref['comparison_payload'],'Full model RMS ON/OFF logits/trace/state/trajectory bytes differ'
                r['exact_reference_bytes_equal']=True
        else:
            if a.mode=='timing':
                assert a.reference,'Correctness report required before full-model speed testing'
                ref=json.loads(Path(a.reference).read_text());assert ref['completed'] and ref.get('exact_reference_bytes_equal')
                assert ref['artifact_id']==r['artifact_id'] and ref['prompt_token_ids']==prompts
            s=model.create_session(a.capacity);sessions.append(s);batch=model.create_batch([a.capacity]*8)
            counts=[a.profile_count] if a.mode=='profile' else ([1,8] if a.trial%2==0 else [8,1])
            for count in counts:
                obj=s if count==1 else batch;inputs=prompts[0] if count==1 else prompts
                warm=obj.generate(inputs,max_new_tokens=a.new_tokens,eos_token_ids=[]);obj.reset()
                first,ttft=measured(lambda:obj.generate(inputs,max_new_tokens=1,eos_token_ids=[]))
                continuation=[first[-1]] if count==1 else [[x[-1]] for x in first]
                if a.mode=='profile':
                    runtime=C.CDLL('/usr/local/cuda/lib64/libcudart.so.12')
                    if a.counter_start:assert runtime.cudaProfilerStart()==0
                    with torch.cuda.nvtx.range('rms_native_post_first'):
                        result,decode=measured(lambda:obj.generate(continuation,max_new_tokens=4,eos_token_ids=[]),4*count)
                    if a.counter_start:assert runtime.cudaProfilerStop()==0
                    r['trials'].append(dict(count=count,profiled=True,decode=decode));continue
                kv_before=model.kv_cache_info()
                result,decode=measured(lambda:obj.generate(continuation,max_new_tokens=a.new_tokens-1,eos_token_ids=[]),(a.new_tokens-1)*count)
                kv_after=model.kv_cache_info()
                split=first+result if count==1 else [x+y for x,y in zip(first,result)]
                obj.reset();whole,unsplit=measured(lambda:obj.generate(inputs,max_new_tokens=a.new_tokens,eos_token_ids=[]),a.new_tokens*count)
                assert whole==split==warm
                expected=ref['comparison_payload']['scalar'][0]['tokens'] if count==1 else ref['comparison_payload']['cohort']['tokens']
                assert whole==expected,'Timed fixed-work trajectory differs from prior correctness gate'
                r['trials'].append(dict(count=count,ttft=ttft,decode=decode,complete=unsplit,output_ids=whole,
                    decode_per_request_tps=(a.new_tokens-1)/decode['wall_seconds'],kv_before_decode=kv_before,kv_after_decode=kv_after,info=obj.info()))
                obj.reset()
        after=Path(a.artifact).stat();assert (after.st_size,after.st_mtime_ns,after.st_ino)==(stat.st_size,stat.st_mtime_ns,stat.st_ino)
        f=model._native.venturi_qwen_experiment_info;f.argtypes=[C.c_void_p,C.POINTER(C.c_uint64)];f.restype=C.c_int
        experiment_info=(C.c_uint64*4)();model._backend._check(f(model._handle,experiment_info));r['experiment_info']=list(experiment_info)
        r['kv_cache']=model.kv_cache_info()
        r['rss_max_kib']=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss
        r['completed']=True;write(out/'report.json',r)
    finally:
        if batch is not None:batch.close()
        for s in sessions:s.close()
        if model is not None:model.close()
        stop.set();thread.join(timeout=8);write(out/'gpu-telemetry.json',samples)


if __name__=='__main__':main()
