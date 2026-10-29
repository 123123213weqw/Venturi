"""Original-FP8 -> experimental RTN Q4 expert checks; synthetic activations only."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_moe import MoEGeometry
from venturi.models.flash_next_q4 import LazyQ4ExpertArchive, decode_q4_fp16, quantize_q4_rtn
from venturi.ops.flash_next_q4 import NativeQ4


def delta(a,b):
    a,b=a.float(),b.float()
    d=a-b
    return dict(max_abs=float(d.abs().max()),ref_peak=float(b.abs().max()),
                relative_l2=float(d.norm()/b.norm().clamp_min(1e-30)),
                cosine=float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','library','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--stage',choices=['qualify','benchmark','profile'],default='qualify')
    p.add_argument('--qualification')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(4021)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(device).total_memory)
    native=NativeQ4(a.library)
    cp=inspect_checkpoint(a.checkpoint);g=MoEGeometry.from_dict(dict(cp.config))
    report=dict(scope=__doc__,stage=a.stage,source='original official FP8 decoded FP16 then requantized; NOT BF16-derived Q4',
                real_model_tps=False,completed=False,records=[],scores=[],library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
                gates=dict(operator_max_abs='0.002+0.0005*ref_peak',operator_relative_l2=.0005,
                           weight_relative_l2=.16,expert_relative_l2=.3,expert_cosine=.95))
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    if a.stage!='qualify':
        proof=json.loads(Path(a.qualification).read_text());assert proof['completed'] and proof['stage']=='qualify'
        assert proof['library_sha256']==report['library_sha256']
    selected=[0,1,7,13,31,63,127,255,383,511] if a.stage=='qualify' else [0]
    for layer in ([0,3] if a.stage=='qualify' else [0]):
        source=BlockFP8ExpertStore(cp,g,f'model.language_model.layers.{layer}.mlp.experts',decode_strategy='vectorized')
        for expert in selected:
            original=source.read(expert,device='cpu',dtype=torch.float16)
            packed=tuple(quantize_q4_rtn(w) for w in original)
            decoded=tuple(decode_q4_fp16(w) for w in packed)
            prepared=tuple(native.prepare(w,device=device) for w in packed)
            dense=tuple(w.to(device) for w in decoded)
            fp8dense=tuple(w.to(device) for w in original)
            for name,q,d,prepared_w,fp8_w in zip(('gate_up','down'),packed,dense,prepared,fp8dense):
                got=native.decode(prepared_w)
                exact=torch.equal(got,d)
                error=delta(d,fp8_w)
                report['records'].append(dict(kind='weight',layer=layer,expert=expert,matrix=name,
                    decoded_gpu_cpu_bits_exact=exact,quantization_error=error,packed_bytes=q.nbytes,fp16_bytes=d.numel()*2))
                save();assert exact and error['relative_l2']<=.16
                if not exact:torch.save(dict(got=got.cpu(),expected=d.cpu()),out/f'decode-failure-{layer}-{expert}-{name}.pth')
                for rows in ([1,8,13] if a.stage!='profile' else [1]):
                    x=(torch.randn(rows,q.in_features,device=device)*.1).half()
                    expected=F.linear(x.float(),d.float())
                    actual=native.linear(x,prepared_w)
                    error=delta(actual,expected)
                    passed=error['max_abs']<=.002+.0005*error['ref_peak'] and error['relative_l2']<=.0005
                    report['records'].append(dict(kind='linear',layer=layer,expert=expert,matrix=name,rows=rows,numerical=error,passed=passed));save()
                    if not passed:
                        torch.save(dict(x=x.cpu(),actual=actual.cpu(),expected=expected.cpu(),codes=q.codes,scales=q.scales),out/f'linear-failure-{layer}-{expert}-{name}-{rows}.pth')
                        raise AssertionError('Packed Q4 execution differs from same-Q4 FP32 dot oracle')
                    if a.stage=='benchmark':
                        functions={'fp8-decoded-fp16-linear':lambda:F.linear(x,fp8_w),
                                   'q4-decoded-fp16-linear':lambda:F.linear(x,d),
                                   'q4-packed-gemv':lambda:native.linear(x,prepared_w),
                                   'q4-decode-then-linear':lambda:F.linear(x,native.decode(prepared_w))}
                        for fn in functions.values():
                            for _ in range(5):fn()
                        for repeat in range(3):
                            order=list(functions) if repeat%2==0 else list(reversed(functions))
                            for mode in order:
                                torch.cuda.synchronize();ct,wall=time.process_time(),time.perf_counter()
                                for _ in range(100):functions[mode]()
                                torch.cuda.synchronize()
                                report['scores'].append(dict(matrix=name,rows=rows,mode=mode,repeat=repeat,
                                    iterations=100,seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-ct))
                                save()
                    if a.stage=='profile':
                        torch.cuda.synchronize()
                        with torch.cuda.nvtx.range('venturi.q4.packed-gemv'):
                            for _ in range(8):native.linear(x,prepared_w)
                        torch.cuda.synchronize()
            for rows in [1,8,13]:
                x=(torch.randn(rows,g.hidden_size,device=device)*.1).half()
                def expert_fn(gu,down):
                    gate,up=F.linear(x,gu).chunk(2,-1)
                    return F.linear(F.silu(gate)*up,down)
                quantized,ref=expert_fn(*dense),expert_fn(*fp8dense)
                error=delta(quantized,ref)
                passed=error['relative_l2']<=.3 and error['cosine']>=.95
                report['records'].append(dict(kind='quantized-expert',layer=layer,expert=expert,rows=rows,numerical=error,passed=passed));save()
                if not passed:
                    torch.save(dict(x=x.cpu(),quantized=quantized.cpu(),reference=ref.cpu()),out/f'quantization-screen-failure-{layer}-{expert}-{rows}.pth')
                    raise AssertionError('RTN Q4 expert component error exceeds preregistered screen; not a task-quality gate')
            del prepared,dense,fp8dense
    report.update(completed=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save();print(json.dumps(dict(completed=True,records=len(report['records']),scores=len(report['scores']))))


if __name__=='__main__':main()
