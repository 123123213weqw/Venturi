"""Record actual GPU output for the intentionally lost small feature."""
import json
from pathlib import Path

import torch
from benchmark_single_a8 import bind,check,idle

torch.set_num_threads(4)
idle()
root = Path('/home/data/wangyue/artifacts/venturi-single-a8-v1')
probe = bind(root/'libsingle_a8_probe.so')
x = torch.zeros(1,32,device='cuda',dtype=torch.float16)
x[0,0]=1;x[0,1]=0.003
q = torch.zeros(1,128,device='cuda',dtype=torch.int8);q[0,1]=127
s = torch.ones(1,4,device='cuda',dtype=torch.float16)
record = dict(scope='Constructed operator counterexample, NOT a production-model quality case.',
              input_f16=x.cpu().tolist(),weight_codes=q.cpu().tolist(),weight_scales_f16=s.cpu().tolist(),
              oracle_f64=float(x[0,1].item())*127,passes={})
for passes in (1,2):
    work = torch.empty(36*passes,device='cuda',dtype=torch.uint8)
    y = torch.empty(1,1,device='cuda',dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    check(probe.venturi_q8_dp4a_quantize(x.data_ptr(),work.data_ptr(),work.numel(),1,32,passes,stream))
    check(probe.venturi_q8_a8_partial_run(work.data_ptr(),work.numel(),q.data_ptr(),s.data_ptr(),y.data_ptr(),1,1,32,128,passes,stream))
    record['passes'][passes] = dict(output=float(y.item()),workspace_bytes=work.cpu().tolist())
assert record['passes'][1]['output']==0
assert abs(record['passes'][2]['output']-record['oracle_f64'])/record['oracle_f64']<1e-5
(root/'small-feature-counterexample.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('passes','input_f16','weight_codes','weight_scales_f16')},indent=2))
print({k:v['output'] for k,v in record['passes'].items()})
