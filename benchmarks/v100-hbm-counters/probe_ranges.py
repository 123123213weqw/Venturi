"""Small admin-counter compatibility probe, not model bandwidth evidence."""
import argparse,json
import torch
p=argparse.ArgumentParser();p.add_argument('--selected',type=int,default=0);p.add_argument('--profiler-api',action='store_true');a=p.parse_args()
x=[torch.ones(262144,device=f'cuda:{i}') for i in range(2)]
y=[v+v for v in x]
for i in range(2):torch.cuda.synchronize(i)
torch.cuda.set_device(a.selected)
if a.profiler_api:torch.cuda.profiler.start()
with torch.cuda.nvtx.range('hbm_probe_range'):
 for _ in range(3):y=[v+v for v in x]
 for i in range(2):torch.cuda.synchronize(i)
if a.profiler_api:torch.cuda.profiler.stop()
print(json.dumps({'selected':a.selected,'values':[float(v[0]) for v in y],'devices':[{'ordinal':i,'name':torch.cuda.get_device_name(i),'uuid':str(getattr(torch.cuda.get_device_properties(i),'uuid',None))} for i in range(2)]}))
