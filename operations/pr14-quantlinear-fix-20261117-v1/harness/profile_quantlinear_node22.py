"""One separately traced real-model C16 decode step, never a TPS sample."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
R=Path(sys.argv[1]);source=R/'change';sys.path.insert(0,str(source/'src'))
p=argparse.ArgumentParser();p.add_argument('--worker',choices=('reference','candidate'));args=p.parse_args(sys.argv[2:])
if args.worker:
    import torch
    from venturi.models import QwenWeights,NativeQwenModel
    from venturi.ops import CudaBackend
    from venturi.models.qwen.native_runtime import QwenRuntime
    from venturi.quantization import codecs
    qualified=json.loads((R/'evidence/sm70-real-v1-report.json').read_text());assert qualified['passed'] and qualified['completed']
    codecs.EXECUTION_QUALIFIED[(7,0)]=set(codecs.EXECUTION_QUALIFIED[(7,0)])|{'q5_g64_fp16'}
    artifact=json.loads((R/'evidence/artifact.json').read_text());build=json.loads((R/'evidence'/('reference-sm70-build.json' if args.worker=='reference' else 'change-sm70-build.json')).read_text())
    torch.set_num_threads(1)
    weights=QwenWeights.open(artifact['path'])
    model=NativeQwenModel.from_q8(weights,QwenRuntime(build['qwen'],CudaBackend(build['ops'])),weight_mode='resident',dtype=torch.float16,device='cuda:0',max_tokens=32,reserve_bytes=512<<20,metadata_mode='sealed',linear_implementation='dequant' if args.worker=='reference' else 'auto')
    batch=model.create_batch([128]*16)
    try:
        prompt=[(7*i+13)%248320 for i in range(16)]
        outputs=batch.generate([prompt]*16,max_new_tokens=[2]*16,eos_token_ids=[])
        tokens=[[v[-1]]for v in outputs]
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:batch.forward(tokens)
        finally:torch.cuda.synchronize();torch.cuda.profiler.stop()
        print(json.dumps(dict(profile_only=True,rows=16,decode_steps=1,implementation=args.worker,tps=None)),flush=True)
    finally:batch.close();model.close()
else:
    evidence=R/'evidence';out=dict(scope=__doc__,tps=None,hardware_counters='not measured: RmProfilingAdminOnly=1; no permission/DCGM changes',profiles={})
    try:
        for kind in ('reference','candidate'):
            prefix=evidence/('profile-'+kind+'-c16')
            command=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','-o',str(prefix),sys.executable,__file__,str(R),'--worker',kind]
            result=subprocess.run(command,cwd=source,stdout=(evidence/('profile-'+kind+'.log')).open('x'),stderr=subprocess.STDOUT,timeout=180)
            record=dict(exit_code=result.returncode,prefix=str(prefix));out['profiles'][kind]=record
            if result.returncode==0:
                stats=subprocess.run(['/usr/local/bin/nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(prefix)+'.nsys-rep'],stdout=(evidence/('profile-'+kind+'-stats.csv')).open('x'),stderr=subprocess.STDOUT,timeout=60)
                record['stats_exit_code']=stats.returncode
            else:break
    except Exception as exc:out['error']=repr(exc)
    finally:(evidence/'profile-diagnostics.json').write_text(json.dumps(out,indent=2))
