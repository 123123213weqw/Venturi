"""Explicit paired KV-append libraries, not a model/TPS benchmark."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true')
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous benchmark evidence')
    libs={};identity={}
    for mode,path in(('baseline',args.baseline),('candidate',args.candidate)):
        lib=C.CDLL(path);lib.cow_bench_run.argtypes=[C.POINTER(C.c_double),C.POINTER(C.c_uint64)]
        lib.cow_bench_run.restype=C.c_int;lib.cow_bench_error.restype=C.c_char_p
        libs[mode]=lib;identity[mode]=dict(file=str(Path(path).resolve()),SHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    def run(mode):
        seconds=(C.c_double*2)();stats=(C.c_uint64*6)()
        begin=time.time();result=libs[mode].cow_bench_run(seconds,stats);end=time.time()
        if result:raise RuntimeError(libs[mode].cow_bench_error().decode())
        g,n,changed,physical,d2h,h2d=stats
        if changed!=(18 if mode=='baseline'else 2):raise RuntimeError('Unexpected actual page identity changes')
        return dict(mode=mode,wall_seconds=seconds[0],CPU_seconds=seconds[1],page_bytes=g,prefix_bytes_per_plane=n,
            changed_pages=changed,physical_pool_bytes=physical,reported_d2h=d2h,reported_h2d=h2d,
            derived_bounce_bytes_both_directions=2*changed*g,unix_started=begin,unix_ended=end,
            unix_window_scope='Entire C function invocation incl setup and byte checks; NOT the returned COW critical-region wall/CPU interval',
            correctness='byte-exact donor/borrower, both planes, verified OUTSIDE timing')
    warmups=[];groups=[]
    for rep in range(3):
        for mode in(('baseline','candidate')if rep%2==0 else('candidate','baseline')):
            warmups.append(dict(rep=rep,**run(mode)));groups.append(dict(rep=rep,**run(mode)))
    report=dict(scope=__doc__,libraries=identity,warmups=warmups,groups=groups,repetitions=3,
        medians={m:{k:statistics.median(r[k]for r in groups if r['mode']==m)for k in('wall_seconds','CPU_seconds','physical_pool_bytes','derived_bounce_bytes_both_directions')}for m in libs},
        model_TPS=None,achieved_SM_occupancy=None,DRAM_bandwidth=None,
        timing='ONLY prepare/admission + ensure/COW + 8-byte append/plane + GPU fence; fresh pool/prefix/map setup and byte checks OUTSIDE',
        limits=['No attention/model/logits/quality/TPS measurement','Original counters omit COW copies; derived bytes are NOT measured bandwidth','3 matched unprofiled repetitions, not a statistical significance claim'],profiles=[])
    # Save unprofiled samples BEFORE requesting an independent profiler pass.
    output.write_text(json.dumps(report,indent=2)+'\n')
    if args.profile:
        import torch
        for mode in libs:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
                run(mode)
            file=output.with_name(output.stem+'-'+mode+'-profile.json')
            if file.exists():raise FileExistsError(file)
            prof.export_chrome_trace(str(file))
            report['profiles'].append(dict(mode=mode,file=file.name,SHA256=hashlib.sha256(file.read_bytes()).hexdigest(),scope='Separate entire mechanism invocation; NEVER used for timed medians'))
        output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in('scope','medians','model_TPS','profiles')},indent=2))


if __name__=='__main__':main()
