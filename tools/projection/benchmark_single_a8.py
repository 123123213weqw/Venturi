"""Single versus residual A8 operator screen, not native model/serving quality.

Exact production Q8_0 weight bytes; synthetic FP16 activations. Both packed
projection and quantize+projection scopes are measured independently, with
unchanged dispatch geometries. All model math/transport outside these operators
is excluded. The production service is never stopped or reconfigured.
"""
import argparse
import ctypes as C
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
from time import monotonic, perf_counter, process_time

import torch

from benchmark_q8_prefetch import GGUF, Descriptor, bind as bind_existing, check, idle, save


CASES = [
    ("mlp-fused", [("blk.0.ffn_gate.weight", slice(0,8704), None), ("blk.0.ffn_up.weight", slice(0,8704), None)]),
    ("mlp-down", [("blk.0.ffn_down.weight", None, (0,8704))]),
    ("gdn-fused", [("blk.0.attn_qkv.weight", list(range(1024))+list(range(2048,3072))+list(range(4096,7168)), None), ("blk.0.attn_gate.weight", slice(0,3072), None)]),
    ("gdn-out", [("blk.0.ssm_out.weight", None, (0,3072))]),
    ("attention-fused", [("blk.3.attn_q.weight", slice(0,6144), None), ("blk.3.attn_k.weight", slice(0,512), None), ("blk.3.attn_v.weight", slice(0,512), None)]),
]
NAMES = ["pinned-two", "local-two", "single-a8"]


def bind(path):
    lib = bind_existing(path)
    for name, args in {
        "venturi_q8_a8_fused_run": [C.c_void_p, C.c_uint64, C.POINTER(Descriptor)] + [C.c_int]*5 + [C.c_void_p],
        "venturi_q8_a8_partial_run": [C.c_void_p, C.c_uint64] + [C.c_void_p]*3 + [C.c_int]*5 + [C.c_void_p],
    }.items():
        if hasattr(lib, name):
            fn = getattr(lib, name);fn.argtypes = args;fn.restype = C.c_int
    return lib


def metrics(actual, reference):
    a, b = actual.double().reshape(-1), reference.double().reshape(-1)
    error = a-b
    ac, bc = a-a.mean(), b-b.mean()
    cosine = float(torch.dot(ac,bc)/(ac.norm()*bc.norm()+1e-30))
    if not bool(ac.any()) and not bool(bc.any()):cosine = 1.0
    rel = float(error.norm()/(b.norm()+1e-30))
    return dict(relative_l2=rel, centered_cosine=cosine,
                max_absolute_error=float(error.abs().max()),
                rms_error=float(error.square().mean().sqrt()),
                changed_projection_argmax_fraction=float((actual.argmax(-1)!=reference.argmax(-1)).double().mean()),
                operator_screen_passed=rel<=0.01 and cosine>=0.9999)


def inputs(tokens, k, family):
    gen = torch.Generator().manual_seed(571+tokens)
    x = torch.randn(tokens,k,generator=gen)*0.25
    if family == "group-outlier":
        x *= 0.006
        cols = torch.arange(0,k,32)
        for t in range(tokens):x[t,cols] = 1.0 if t%2 else -1.0
    elif family == "mixed-range":
        x *= torch.logspace(-3,3,(k+31)//32).repeat_interleave(32)[:k]
    else:assert family == "gaussian"
    return x.half()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--control-library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--tokens", default="1,2,4,8,23,32")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--profile-case", choices=[v[0] for v in CASES])
    p.add_argument("--profile-arm", choices=NAMES, default="local-two")
    a = p.parse_args()
    out = Path(a.output_directory);out.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.get_device_capability() == (7,0) and a.repeats>=3
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32 = False
    idle();assert torch.cuda.mem_get_info()[0]>2*2**30
    probe, control = bind(a.library), bind(a.control_library)
    gguf = GGUF(a.gguf)
    tokens = [int(t) for t in a.tokens.split(',')]
    assert tokens and all(1<=t<=32 for t in tokens)
    stat = Path(a.gguf).stat()
    report = dict(scope=__doc__, args=vars(a), utc=datetime.now(timezone.utc).isoformat(),
                  completed=False,profiled=bool(a.profile_case), cases=[],
                  gguf=dict(path=a.gguf,architecture=gguf.meta['general.architecture'],size=stat.st_size,mtime_ns=stat.st_mtime_ns),
                  library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
                  control_library_sha256=hashlib.sha256(Path(a.control_library).read_bytes()).hexdigest(),
                  device_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  precision='Identical group32 Q8 weight codes/F16 scales; F16 input, A8 scales F32. Single A8 removes residual pass and changes arithmetic; F16 fused/F32 TP partial outputs.',
                  quality_scope='Full operator outputs versus F32 represented-Q8-weight/F16-activation reference, with sampled independent F64 reference. Synthetic inputs are NOT production model activation or corpus quality.')
    stop, samples = threading.Event(), []

    def sample():
        while not stop.is_set():
            try:
                r = subprocess.run(['nvidia-smi','-i','0','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                samples.append(dict(monotonic=monotonic(),returncode=r.returncode,csv=r.stdout.strip()))
            except Exception as e:samples.append(dict(monotonic=monotonic(),error=repr(e)))
            stop.wait(0.2)
    thread = threading.Thread(target=sample);thread.start()

    def run_case(label, specs):
        idle()
        cpu_weights = [gguf.q8(*s) for s in specs]
        weights = [(q.cuda(),s.cuda()) for q,s in cpu_weights]
        pins = [dict(codes_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest(),
                     scales_sha256=hashlib.sha256(s.numpy().tobytes()).hexdigest()) for q,s in cpu_weights]
        partial = len(weights)==1
        k, ns = weights[0][0].shape[1], [q.shape[0] for q,s in weights]
        for t in tokens:
            idle()
            xcpu = inputs(t,k,'gaussian');x = xcpu.cuda()
            count = t*((k+31)//32)
            work2 = torch.empty(count*72,device='cuda',dtype=torch.uint8)
            work1 = torch.empty(count*36,device='cuda',dtype=torch.uint8)
            workspaces = [work2,work2,work1]
            passes = [2,2,1]
            ys = [[torch.empty(t,n,device='cuda',dtype=torch.float32 if partial else torch.float16) for n in ns] for name in NAMES]
            desc = [(Descriptor*len(ns))(*[Descriptor(q.data_ptr(),s.data_ptr(),y.data_ptr(),n,n) for (q,s),y,n in zip(weights,outputs,ns)]) for outputs in ys]
            stream = torch.cuda.current_stream().cuda_stream

            def quant(i):
                w = workspaces[i]
                check(probe.venturi_q8_dp4a_quantize(x.data_ptr(),w.data_ptr(),w.numel(),t,k,passes[i],stream))

            def project(i):
                w = workspaces[i];lib = control if i==0 else probe
                if partial:
                    args = [w.data_ptr(),w.numel(),weights[0][0].data_ptr(),weights[0][1].data_ptr(),ys[i][0].data_ptr(),t,ns[0],k,k]
                    fn = lib.venturi_q8_projection_partial_run if i==0 else lib.venturi_q8_a8_partial_run
                    if i:args.append(passes[i])
                else:
                    args = [w.data_ptr(),w.numel(),desc[i],len(ns),t,k,k,passes[i]]
                    fn = lib.venturi_q8_projection_fused_run if i==0 else lib.venturi_q8_a8_fused_run
                check(fn(*args,stream))

            def correctness_and_quality(family):
                xcpu.copy_(inputs(t,k,family));x.copy_(xcpu)
                for i in (0,1,2):quant(i);project(i)
                torch.cuda.synchronize()
                assert all(torch.equal(u.view(torch.uint8),v.view(torch.uint8)) for u,v in zip(ys[0],ys[1]))
                assert torch.equal(work1[:count*32],work2[:count*32])
                assert torch.equal(work1[count*32:],work2[count*64:count*68])
                values = [[y.cpu() for y in outputs] for outputs in ys]
                result = []
                from q8_dp4a_reference import projection_reference
                for j,((q,s),(qcpu,scpu)) in enumerate(zip(weights,cpu_weights)):
                    wfloat = q.float().view(q.shape[0],-1,32)
                    wfloat.mul_(s.float().unsqueeze(-1))
                    ref = (x.float() @ wfloat.reshape(q.shape).T).cpu()
                    del wfloat
                    rows = sorted(set([0,len(qcpu)-1]+torch.randint(len(qcpu),(24,),generator=torch.Generator().manual_seed(62)).tolist()))
                    weight64 = qcpu[rows].double()*scpu[rows].double().repeat_interleave(32,-1)
                    ideal64 = xcpu.double() @ weight64.T
                    assert float((ref[:,rows].double()-ideal64).norm()/(ideal64.norm()+1e-30))<1e-5
                    implementation_errors = {}
                    for i in (1,2):
                        represented = projection_reference(xcpu,qcpu[rows],scpu[rows],passes[i])
                        error = float((values[i][j][:,rows].double()-represented).norm()/(represented.norm()+1e-30))
                        assert error<0.001
                        implementation_errors[NAMES[i]] = error
                    result.append(dict(rows=ns[j],family=family,implementation_fp64_sample_relative_l2=implementation_errors,
                                       versus_unquantized_activation={name:metrics(v[j],ref) for name,v in zip(NAMES,values)},
                                       single_versus_two=metrics(values[2][j],values[0][j])))
                return result

            case = dict(name=label,tokens=t,k=k,rows=ns,weight_pins=pins,partial_fp32=partial,
                        complete_two_pass_bytes_equal=True,one_pass_first_plane_bytes_equal=True,
                        quality=correctness_and_quality('gaussian'),waves=[])
            # Reference setup is outside all timed/profiled ranges.
            gc.collect();torch.cuda.empty_cache()
            if a.profile_case:
                quant(NAMES.index(a.profile_arm));torch.cuda.synchronize()
                check(probe.venturi_a8_profile_start())
                for _ in range(16):project(NAMES.index(a.profile_arm))
                torch.cuda.synchronize();check(probe.venturi_a8_profile_stop())
            else:
                for scope in ('packed','quantize+projection'):
                    graphs = []
                    for i in range(3):
                        side = torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=side):
                            stream = side.cuda_stream
                            for _ in range(20):
                                if scope=='quantize+projection':quant(i)
                                project(i)
                        torch.cuda.current_stream().wait_stream(side);graphs.append(graph)
                    stream = torch.cuda.current_stream().cuda_stream
                    for graph in graphs:
                        for _ in range(5):graph.replay()
                    torch.cuda.synchronize()
                    for repeat in range(a.repeats):
                        idle()
                        for i in (range(3) if repeat%2==0 else reversed(range(3))):
                            torch.cuda.synchronize()
                            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            wall,cpu = perf_counter(),process_time();start.record()
                            for _ in range(5):graphs[i].replay()
                            end.record();end.synchronize()
                            case['waves'].append(dict(scope=scope,arm=NAMES[i],repeat=repeat,calls=100,
                                                      gpu_event_ms_per_call=start.elapsed_time(end)/100,
                                                      wall_seconds=perf_counter()-wall,process_cpu_seconds=process_time()-cpu))
                    del graphs,graph,side
                case['median_ms'] = {scope:{name:statistics.median(w['gpu_event_ms_per_call'] for w in case['waves'] if w['scope']==scope and w['arm']==name) for name in NAMES} for scope in ('packed','quantize+projection')}
                if t==8:
                    for family in ('group-outlier','mixed-range'):case['quality'].extend(correctness_and_quality(family))
            case['torch_reserved_bytes'] = torch.cuda.memory_reserved()
            case['torch_allocated_bytes'] = torch.cuda.memory_allocated()
            assert case['torch_reserved_bytes']<1024**3
            report['cases'].append(case);save(out/'report.json',report)
            print(json.dumps({key:case[key] for key in ('name','tokens','median_ms') if key in case}),flush=True)
            del quant,project,correctness_and_quality,ys,desc,workspaces,work1,work2,x,xcpu
            gc.collect();torch.cuda.empty_cache()

    try:
        for label,specs in CASES:
            if not a.profile_case or label==a.profile_case:run_case(label,specs)
        assert Path(a.gguf).stat().st_mtime_ns==stat.st_mtime_ns
        report['completed'] = True;save(out/'report.json',report)
    finally:
        stop.set();thread.join(timeout=8);save(out/'gpu-telemetry.json',samples)


if __name__ == '__main__':main()
