"""Same-bitplane, bit-identical row decode microbenchmark; NOT model TPS."""
import ctypes as C
import hashlib,json,os,sys,time
from pathlib import Path
import torch
from venturi.ops import CudaBackend
r=Path(sys.argv[1]);e=r/'evidence';out=e/'row-decode-micro-v1.json';assert not out.exists()
new=json.loads((e/'change-sm89-build.json').read_text())['ops']
old='/data/wangyue/codex-build/Venturi-pr14-fix-20261117/native/change-sm89-ops/a3300053cc6d611c/libventuri_ops.so'
backends={k:CudaBackend(v)for k,v in [('reference',old),('candidate',new)]}
fns={k:b._native.venturi_row_dequant_rows for k,b in backends.items()}
for f in fns.values():f.argtypes=[C.c_void_p]*4+[C.c_int]*5+[C.c_void_p];f.restype=C.c_int
torch.manual_seed(79);torch.set_num_threads(1)
report=dict(scope=__doc__,architecture=list(torch.cuda.get_device_capability()),libraries={k:dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for k,p in [('reference',old),('candidate',new)]},samples=[],timing='CUDA events, 16 decodes per sample after 8 warmups; 3 repetitions per shape/format/family. CPU process and common wall measured separately; no profiler. Repeated input is hot, not cold weights/model inference.',hardware_counters='not measured; GPU busy, VRAM, power, clocks and temperature captured by outer guard; no occupancy/bandwidth counter claim')
stream=torch.cuda.current_stream().cuda_stream
for n,k in [(8192,5120),(5120,17408)]:
 for bits in (4,5,6):
  padded=(k+127)//128*128
  codes=torch.randint(0,256,(n,padded//2),dtype=torch.uint8,device='cuda')
  high=torch.randint(0,256,(n,padded*(bits-4)//8),dtype=torch.uint8,device='cuda') if bits>4 else None
  scales=(torch.rand(n,padded//64,device='cuda')*.02).half()
  y={tag:torch.empty((n,k),device='cuda',dtype=torch.float16)for tag in fns}
  def invoke(tag):
   rc=fns[tag](codes.data_ptr(),high.data_ptr()if high is not None else None,scales.data_ptr(),y[tag].data_ptr(),0,n,k,padded,bits,stream)
   if rc:raise RuntimeError(f'row decoder failed: {rc}')
  for tag in fns:invoke(tag)
  torch.cuda.synchronize();assert torch.equal(y['reference'].view(torch.uint8),y['candidate'].view(torch.uint8))
  for rep in range(3):
   # Alternating order reduces systematic thermal/order bias.
   for tag in (('reference','candidate') if rep%2==0 else ('candidate','reference')):
    for _ in range(8):invoke(tag)
    torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    wall=time.perf_counter();cpu=time.process_time();a.record()
    for _ in range(16):invoke(tag)
    b.record();b.synchronize()
    row=dict(n=n,k=k,bits=bits,repeat=rep,implementation=tag,iterations=16,cuda_ms_per_decode=a.elapsed_time(b)/16,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,bit_identical_to_frozen_reference=True)
    report['samples'].append(row);out.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
report['completed']=True;out.write_text(json.dumps(report,indent=2))
