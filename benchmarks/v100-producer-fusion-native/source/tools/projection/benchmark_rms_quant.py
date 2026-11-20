"""Research RMSNorm/two-pass A8 fusion; real Qwen3.8 gate/up slices, synthetic inputs and norm weights.

Producer and producer+gate/up only. Not native integration, full-model or API TPS.
All timed arms share exact workspace/output addresses and pinned projection kernels.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
from time import monotonic,perf_counter,process_time

import torch
from benchmark_q8_prefetch import GGUF,Descriptor,bind,check,idle,save

NAMES=['separate','chunk8','chunk32','chunk128']
SCOPES=['producer','producer+gate-up']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['gguf','library','control-library','output-directory']:p.add_argument('--'+name,required=True)
    p.add_argument('--tokens',default='1,2,4,8,23,32');p.add_argument('--layers',default='0,31,63')
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--profile-arm',choices=NAMES)
    a=p.parse_args();out=Path(a.output_directory);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);assert torch.cuda.get_device_capability()==(7,0) and a.repeats>=3
    idle();assert torch.cuda.mem_get_info()[0]>2*2**30
    q,c=C.CDLL(a.library),bind(a.control_library)
    q.venturi_rms_q8_quantize.argtypes=[C.c_void_p]*3+[C.c_uint64,C.c_int,C.c_int,C.c_float,C.c_int,C.c_void_p,C.c_void_p];q.venturi_rms_q8_quantize.restype=C.c_int
    c.venturi_rms_run.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_float,C.c_void_p];c.venturi_rms_run.restype=C.c_int
    gguf=GGUF(a.gguf);stamp=Path(a.gguf).stat()
    report=dict(scope=__doc__,completed=False,profiled=bool(a.profile_arm),args=vars(a),cases=[],
        device_uuid=str(torch.cuda.get_device_properties(0).uuid),library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        control_library_sha256=hashlib.sha256(Path(a.control_library).read_bytes()).hexdigest(),
        precision='F16 inputs and preserved norm rounding; same original 256-thread FP32 norm reduction, group32 two-pass residual A8/F32 scales, Q8/F16-scale gate/up weights, F16 outputs.',
        gguf=dict(path=a.gguf,size=stamp.st_size,mtime_ns=stamp.st_mtime_ns,architecture=gguf.meta['general.architecture']))
    stop,samples=threading.Event(),[]
    def sample():
        while not stop.is_set():
            try:
                r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                samples.append(dict(monotonic=monotonic(),returncode=r.returncode,csv=r.stdout.strip()))
            except Exception as e:samples.append(dict(monotonic=monotonic(),error=repr(e)))
            stop.wait(0.2)
    sampler=threading.Thread(target=sample);sampler.start()
    try:
        for layer in map(int,a.layers.split(',')):
            idle();cpu=[gguf.q8(f'blk.{layer}.ffn_{name}.weight',slice(0,8704)) for name in ['gate','up']]
            weights=[(x.cuda(),s.cuda()) for x,s in cpu]
            pins=[dict(codes=hashlib.sha256(x.numpy().tobytes()).hexdigest(),scales=hashlib.sha256(s.numpy().tobytes()).hexdigest()) for x,s in cpu]
            for t in map(int,a.tokens.split(',')):
                assert 1<=t<=32;idle();k,n=5120,8704;size=t*(k//32)*72
                g=torch.Generator().manual_seed(671+t)
                x=(torch.randn(t,k,generator=g)*0.25).half().cuda();w=(1+0.03*torch.randn(k,generator=g)).cuda()
                mids=[torch.empty_like(x) for _ in NAMES];work=[torch.empty(size,device='cuda',dtype=torch.uint8) for _ in NAMES]
                outputs=[[torch.empty(t,n,device='cuda',dtype=torch.float16) for _ in range(2)] for _ in NAMES]
                desc=[(Descriptor*2)(*[Descriptor(codes.data_ptr(),s.data_ptr(),y.data_ptr(),n,n) for (codes,s),y in zip(weights,ys)]) for ys in outputs]
                stream=torch.cuda.current_stream().cuda_stream;shared=False
                def producer(i,debug=False):
                    j=0 if shared else i
                    if i==0:
                        check(c.venturi_rms_run(x.data_ptr(),w.data_ptr(),mids[j].data_ptr(),t,k,0,1e-6,stream))
                        check(c.venturi_q8_dp4a_quantize(mids[j].data_ptr(),work[j].data_ptr(),size,t,k,2,stream))
                    else:check(q.venturi_rms_q8_quantize(x.data_ptr(),w.data_ptr(),work[j].data_ptr(),size,t,k,1e-6,[0,8,32,128][i],mids[j].data_ptr() if debug else None,stream))
                def call(i,scope):
                    producer(i)
                    if scope!='producer':
                        j=0 if shared else i;check(c.venturi_q8_projection_fused_run(work[j].data_ptr(),size,desc[j],2,t,k,k,2,stream))
                for i in range(4):call(i,'producer+gate-up')
                torch.cuda.synchronize();assert all(torch.equal(work[0],v) for v in work[1:])
                assert all(torch.equal(outputs[0][j].view(torch.uint8),outputs[i][j].view(torch.uint8)) for i in range(1,4) for j in range(2))
                for i in range(1,4):producer(i,True)
                torch.cuda.synchronize();assert all(torch.equal(mids[0].view(torch.uint8),v.view(torch.uint8)) for v in mids[1:])
                case=dict(layer=layer,tokens=t,k=k,n=n,synthetic_effective_norm_weights=True,weight_pins=pins,
                    full_workspace_bytes_equal=True,full_projection_bytes_equal=True,full_f16_intermediate_bytes_equal=True,
                    timed_arms_share_addresses=True,waves=[])
                shared=True
                if a.profile_arm:
                    i=NAMES.index(a.profile_arm);check(q.venturi_rq_profile_start());producer(i);torch.cuda.synchronize();check(q.venturi_rq_profile_stop())
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
                            torch.cuda.synchronize();assert torch.equal(work[0],work[1])
                            assert all(torch.equal(outputs[0][j].view(torch.uint8),outputs[1][j].view(torch.uint8)) for j in range(2))
                        for repeat in range(a.repeats):
                            idle()
                            for i in (range(4) if repeat%2==0 else reversed(range(4))):
                                torch.cuda.synchronize();s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                wall,cpu_t=perf_counter(),process_time();s.record()
                                for _ in range(5):graphs[i].replay()
                                e.record();e.synchronize()
                                case['waves'].append(dict(scope=scope,arm=NAMES[i],repeat=repeat,calls=100,gpu_event_ms_per_call=s.elapsed_time(e)/100,wall_seconds=perf_counter()-wall,process_cpu_seconds=process_time()-cpu_t))
                        del graphs,graph,side
                    case['median_ms']={scope:{name:statistics.median(r['gpu_event_ms_per_call'] for r in case['waves'] if r['scope']==scope and r['arm']==name) for name in NAMES} for scope in SCOPES}
                case['torch_reserved_bytes']=torch.cuda.memory_reserved();assert case['torch_reserved_bytes']<1024**3
                report['cases'].append(case);save(out/'report.json',report)
                print(json.dumps({k:case[k] for k in ['layer','tokens','median_ms'] if k in case}),flush=True)
                del producer,call,mids,work,outputs,desc,x,w
                torch.cuda.empty_cache()
        assert Path(a.gguf).stat().st_mtime_ns==stamp.st_mtime_ns
        report['completed']=True;save(out/'report.json',report)
    finally:stop.set();sampler.join(timeout=8);save(out/'gpu-telemetry.json',samples)


if __name__=='__main__':main()
