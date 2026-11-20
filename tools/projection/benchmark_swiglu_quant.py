"""Bit-preserving SwiGLU + residual-A8 producer fusion on SM70.

Measures producer, producer+Down, and a single TP-rank MLP projection chain.
Uses production Qwen3.8 Q8_0 weights and synthetic F16 hidden inputs; not actual
full-model activations or serving TPS. No production/default dispatch changes.
SwiGLU is out-of-place for the repeated separate baseline; input tensors are
not silently modified between replays. All arms share pinned projection kernels.
"""
import argparse
import ctypes as C
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
from time import monotonic,perf_counter,process_time

import torch

from benchmark_q8_prefetch import GGUF,Descriptor,bind as old_bind,check,idle,save

NAMES=['pinned-separate','local-separate','fused-128','fused-256']
SCOPES=['producer','producer+down','mlp-shard']


def bind(path):
    lib=old_bind(path)
    for name,args in {
        'venturi_swiglu_run':[C.c_void_p]*3+[C.c_int64,C.c_int,C.c_void_p],
        'venturi_swiglu_clone_run':[C.c_void_p]*3+[C.c_int64,C.c_int,C.c_void_p],
        'venturi_swiglu_q8_quantize':[C.c_void_p]*3+[C.c_uint64]+[C.c_int]*3+[C.c_void_p]*2,
    }.items():
        if hasattr(lib,name):
            fn=getattr(lib,name);fn.argtypes=args;fn.restype=C.c_int
    return lib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--control-library',required=True)
    p.add_argument('--output-directory',required=True)
    p.add_argument('--tokens',default='1,2,4,8,23,32')
    p.add_argument('--layers',default='0,31,63')
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--profile-arm',choices=NAMES)
    a=p.parse_args()
    out=Path(a.output_directory);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    assert torch.cuda.get_device_capability()==(7,0) and a.repeats>=3
    idle();assert torch.cuda.mem_get_info()[0]>2*2**30
    probe,control=bind(a.library),bind(a.control_library)
    gguf=GGUF(a.gguf);stat=Path(a.gguf).stat()
    tokens=[int(t) for t in a.tokens.split(',')];layers=[int(v) for v in a.layers.split(',')]
    assert tokens and all(1<=t<=32 for t in tokens) and layers and all(0<=v<64 for v in layers)
    report=dict(scope=__doc__,args=vars(a),utc=datetime.now(timezone.utc).isoformat(),
        completed=False,profiled=bool(a.profile_arm),cases=[],
        precision='Same Q8 codes/F16 weight scales; stable SwiGLU rounded to F16; group32 TWO-pass residual A8, F32 activation scales/Down partial outputs. No single-A8 arithmetic.',
        device_uuid=str(torch.cuda.get_device_properties(0).uuid),
        gguf=dict(path=a.gguf,size=stat.st_size,mtime_ns=stat.st_mtime_ns,architecture=gguf.meta['general.architecture']),
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        control_library_sha256=hashlib.sha256(Path(a.control_library).read_bytes()).hexdigest(),
        kernels_per_logical_call={
            'producer':{n:2 if i<2 else 1 for i,n in enumerate(NAMES)},
            'producer+down':{n:3 if i<2 else 2 for i,n in enumerate(NAMES)},
            'mlp-shard':{n:5 if i<2 else 4 for i,n in enumerate(NAMES)}})
    stop,samples=threading.Event(),[]
    def sample():
        while not stop.is_set():
            try:
                s=subprocess.run(['nvidia-smi','-i','0','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                samples.append(dict(monotonic=monotonic(),returncode=s.returncode,csv=s.stdout.strip()))
            except Exception as e:samples.append(dict(monotonic=monotonic(),error=repr(e)))
            stop.wait(0.2)
    thread=threading.Thread(target=sample);thread.start()

    def run_layer(layer):
        idle();prefix=f'blk.{layer}.ffn_'
        cpu=[gguf.q8(prefix+'gate.weight',slice(0,8704)),gguf.q8(prefix+'up.weight',slice(0,8704)),gguf.q8(prefix+'down.weight',None,(0,8704))]
        weights=[(q.cuda(),s.cuda()) for q,s in cpu]
        pins=[dict(codes_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest(),scales_sha256=hashlib.sha256(s.numpy().tobytes()).hexdigest()) for q,s in cpu]
        for t in tokens:
            idle();k,n,hidden=8704,5120,5120
            gen=torch.Generator().manual_seed(571+t)
            x=(torch.randn(t,hidden,generator=gen)*0.25).half().cuda()
            gate,up=[torch.empty(t,k,device='cuda',dtype=torch.float16) for _ in range(2)]
            mids=[torch.empty_like(gate) for _ in range(2)]
            size=t*((k+31)//32)*72
            work=[torch.empty(size,device='cuda',dtype=torch.uint8) for _ in NAMES]
            input_work=torch.empty(t*(hidden//32)*72,device='cuda',dtype=torch.uint8)
            outputs=[torch.empty(t,n,device='cuda',dtype=torch.float32) for _ in NAMES]
            desc=(Descriptor*2)(*[Descriptor(q.data_ptr(),s.data_ptr(),y.data_ptr(),k,k) for (q,s),y in zip(weights[:2],(gate,up))])
            stream=torch.cuda.current_stream().cuda_stream
            timed_shared_buffers=False

            def gu():
                check(control.venturi_q8_dp4a_quantize(x.data_ptr(),input_work.data_ptr(),input_work.numel(),t,hidden,2,stream))
                check(control.venturi_q8_projection_fused_run(input_work.data_ptr(),input_work.numel(),desc,2,t,hidden,hidden,2,stream))

            def producer(i):
                j=0 if timed_shared_buffers else i
                if i<2:
                    lib=control if i==0 else probe
                    fn=lib.venturi_swiglu_run if i==0 else lib.venturi_swiglu_clone_run
                    mid=mids[0 if timed_shared_buffers else i]
                    check(fn(gate.data_ptr(),up.data_ptr(),mid.data_ptr(),t*k,0,stream))
                    check(lib.venturi_q8_dp4a_quantize(mid.data_ptr(),work[j].data_ptr(),size,t,k,2,stream))
                else:
                    check(probe.venturi_swiglu_q8_quantize(gate.data_ptr(),up.data_ptr(),work[j].data_ptr(),size,t,k,128 if i==2 else 256,None,stream))

            def down(i):
                j=0 if timed_shared_buffers else i
                q,s=weights[2]
                check(control.venturi_q8_projection_partial_run(work[j].data_ptr(),size,q.data_ptr(),s.data_ptr(),outputs[j].data_ptr(),t,n,k,k,stream))

            def call(i,scope):
                if scope=='mlp-shard':gu()
                producer(i)
                if scope!='producer':down(i)

            gu()
            for i in range(4):call(i,'producer+down')
            torch.cuda.synchronize()
            assert all(torch.equal(work[0],w) for w in work[1:]),'Encoded two-pass activation bytes changed'
            assert torch.equal(mids[0].view(torch.uint8),mids[1].view(torch.uint8)),'Clone F16 arithmetic drift'
            assert all(torch.equal(outputs[0].view(torch.uint8),y.view(torch.uint8)) for y in outputs[1:]),'Down output bytes changed'
            from q8_dp4a_reference import projection_reference
            rows=sorted(set([0,n-1]+torch.randint(n,(24,),generator=torch.Generator().manual_seed(62)).tolist()))
            ideal=projection_reference(mids[0].cpu(),cpu[2][0][rows],cpu[2][1][rows],2)
            error=float((outputs[0][:,rows].cpu().double()-ideal).norm()/(ideal.norm()+1e-30));assert error<1e-5
            case=dict(layer=layer,tokens=t,hidden=hidden,intermediate_per_rank=k,down_rows=n,weight_pins=pins,
                      complete_workspace_bytes_equal=True,complete_down_bytes_equal=True,clone_f16_bytes_equal=True,
                      sampled_independent_fp64_down_relative_l2=error,
                      timed_arms_share_intermediate_workspace_and_down_output_addresses=True,waves=[])
            # Isolate producer fusion from allocation/cache-placement changes in
            # the common Down kernel. Raw correctness above used separate outputs.
            timed_shared_buffers=True
            if a.profile_arm:
                i=NAMES.index(a.profile_arm)
                check(probe.venturi_sq_profile_start());producer(i);torch.cuda.synchronize();check(probe.venturi_sq_profile_stop())
            else:
                for scope in SCOPES:
                    graphs=[]
                    for i in range(4):
                        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=side):
                            stream=side.cuda_stream
                            for _ in range(20):call(i,scope)
                        torch.cuda.current_stream().wait_stream(side);graphs.append(graph)
                    stream=torch.cuda.current_stream().cuda_stream
                    for graph in graphs:
                        for _ in range(5):graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal(work[0],work[1])
                        assert torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8))
                    torch.cuda.synchronize()
                    for repeat in range(a.repeats):
                        idle()
                        for i in (range(4) if repeat%2==0 else reversed(range(4))):
                            torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            wall,cput=perf_counter(),process_time();start.record()
                            for _ in range(5):graphs[i].replay()
                            end.record();end.synchronize()
                            case['waves'].append(dict(scope=scope,arm=NAMES[i],repeat=repeat,calls=100,
                                gpu_event_ms_per_call=start.elapsed_time(end)/100,
                                wall_seconds=perf_counter()-wall,process_cpu_seconds=process_time()-cput))
                    del graphs,graph,side
                case['median_ms']={scope:{name:statistics.median(w['gpu_event_ms_per_call'] for w in case['waves'] if w['scope']==scope and w['arm']==name) for name in NAMES} for scope in SCOPES}
                # Re-run after all captured graphs; also verify their outputs.
                assert all(torch.equal(work[0],w) for w in work[1:])
                assert all(torch.equal(outputs[0].view(torch.uint8),y.view(torch.uint8)) for y in outputs[1:])
            case['torch_reserved_bytes']=torch.cuda.memory_reserved();case['torch_allocated_bytes']=torch.cuda.memory_allocated()
            assert case['torch_reserved_bytes']<1024**3
            report['cases'].append(case);save(out/'report.json',report)
            print(json.dumps({k:case[k] for k in ('layer','tokens','median_ms') if k in case}),flush=True)
            del gu,producer,down,call,work,input_work,outputs,mids,x,gate,up,desc
            gc.collect();torch.cuda.empty_cache()
    try:
        for layer in layers:run_layer(layer)
        assert Path(a.gguf).stat().st_mtime_ns==stat.st_mtime_ns
        report['completed']=True;save(out/'report.json',report)
    finally:
        stop.set();thread.join(timeout=8);save(out/'gpu-telemetry.json',samples)


if __name__=='__main__':main()
