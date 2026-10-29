"""Separate Q4 packed-storage/transient-GEMM trial, NOT packed GEMV promotion.

Requires the first trial's complete32-step CANONICAL Q4 quality screen, even
though that trial's packed GEMV execution failed. New independent CPU-vs-GPU Q4
decode/full-state checks precede timing. No relaxed packed-kernel gate or native
full-model claim. Source remains FP8-requantized, PLE/core precision unchanged.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from benchmark_flash_next_full import checked_payloads,process_memory
from benchmark_flash_next_q4_full import PROMPT,write
from flash_next_hybrid_metrics import difference,state_to_cpu,compare_state
from venturi.checkpoint import inspect_checkpoint
from venturi.flash_next_generation import generate_flash_next_text
from venturi.generation import LocalTokenizer
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_moe import MoEGeometry
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_q4 import LazyQ4ExpertArchive,PackedQ4Dispatch
from venturi.models.flash_next_request import FlashNextRequest
from venturi.ops.flash_next_q4 import NativeQ4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','fp8-library','q4-library','output','archive-root','quality-proof']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--stage',choices=['correctness','benchmark','profile'],required=True)
    p.add_argument('--qualification')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');assert torch.cuda.get_device_capability(device)==(7,0)
    torch.cuda.set_per_process_memory_fraction(.85)
    proof,manifest=checked_payloads(Path(a.checkpoint));cp=inspect_checkpoint(a.checkpoint)
    quality=json.loads(Path(a.quality_proof).read_text())
    libraries={key:hashlib.sha256(Path(value).read_bytes()).hexdigest() for key,value in [('fp8',a.fp8_library),('q4',a.q4_library)]}
    if quality['manifest_sha256']!=proof['manifest_sha256'] or quality['libraries']!=libraries or (
            not quality.get('quantization_screen_passed') or len(quality.get('quality_checks',[]))!=32):
        raise ValueError('Completed matching CANONICAL Q4 quality trace required; failed packed execution is not a timing permission')
    start=time.perf_counter()
    loaded=load_weight_decoded_fp8_text(cp,max_core_bytes=12*2**30,max_expert_bytes=64*2**20,max_sequence=256,
        attention_backend='sdpa',execution_device='cuda:0',execution_dtype=torch.float16,max_execution_core_bytes=12*2**30)
    core=loaded.core;torch.cuda.synchronize();g=MoEGeometry.from_dict(dict(cp.config))
    tokenizer=LocalTokenizer(Path(a.checkpoint)/'tokenizer.json',vocab_size=cp.config['text_config']['vocab_size'])
    native=NativeQ4(a.q4_library)
    archives=[LazyQ4ExpertArchive(BlockFP8ExpertStore(cp,g,f'model.language_model.layers.{i}.mlp.experts',decode_strategy='vectorized'),
        Path(a.archive_root)/f'layer-{i:02d}',source_identity=dict(revision=manifest['hf_revision'],
        manifest_sha256=proof['manifest_sha256'],layer=i,decode_dtype='FP16')) for i in range(len(core.layers))]
    report=dict(scope=__doc__,stage=a.stage,manifest_sha256=proof['manifest_sha256'],libraries=libraries,
        quality_proof_sha256=hashlib.sha256(Path(a.quality_proof).read_bytes()).hexdigest(),
        quality_summary=quality['quality_summary'],previous_packed_execution_failed=not quality['execution_passed'],
        execution='ALL expert GEMMs Torch FP16; Q4 retained packed, GPU decode only transient',
        native_packed_gemv_used=False,source='Original FP8 decoded FP16 then RTN Q4, NOT BF16-derived',
        prompt=PROMPT,prompt_ids=tokenizer.encode(PROMPT),output_tokens=32,decode_transitions=31,prefill_chunk=16,
        context_reservation=256,concurrency=1,loading_seconds=time.perf_counter()-start,records=[],
        complete_q4_checkpoint=False,native_full_model=False,completed=False,throughput_scored=False)
    dispatches=[]
    def save():write(out/'report.json',report)
    def clear():
        nonlocal dispatches
        for d in dispatches:d.clear()
        dispatches=[]
        for layer in core.layers:
            layer.mlp.expert_dispatch=None
            if hasattr(layer.mlp.expert_store,'clear'):layer.mlp.expert_store.clear()
        gc.collect()
    def activate(mode):
        clear()
        if mode=='fp8-gpu-lru':
            return bind_fp8_expert_policy(core,cp,backend='cuda-byte-decode',retained_bytes_per_layer=(256*2**20,)*len(core.layers),library=a.fp8_library)
        for layer,archive in zip(core.layers,archives):
            d=PackedQ4Dispatch(archive,native,device=device,max_resident_bytes=256*2**20)
            dispatches.append(d);layer.mlp.expert_store=archive;layer.mlp.expert_dispatch=d.decoded_reference
        return dict(mode=mode,packed_budget=256*2**20*len(core.layers),retained_decoded_bytes=0)
    def stats():
        return [d.stats for d in dispatches] if dispatches else [layer.mlp.expert_store.stats for layer in core.layers]
    save()
    try:
        with PreparedNGramTable(next(iter(loaded.tables.values()))) as reader:
            if a.stage=='correctness':
                activate('q4-transient-gemm')
                cpu_request,gpu_request=FlashNextRequest(core,reader),FlashNextRequest(core,reader)
                checks=[]
                with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                    for step,ids in enumerate(([248045],[846],[198])):
                        for layer in core.layers:layer.mlp.expert_dispatch=None
                        p=cpu_request.preview(ids);cpu_request.commit(p);expected=p.logits.cpu();expected_state=state_to_cpu(cpu_request.state)
                        for layer,d in zip(core.layers,dispatches):layer.mlp.expert_dispatch=d.decoded_reference
                        q=gpu_request.preview(ids);gpu_request.commit(q);actual=q.logits.cpu();actual_state=state_to_cpu(gpu_request.state)
                        error=difference(actual,expected,kind='logits');states=compare_state(actual_state,expected_state)
                        exact=error['exact'] and all(x['exact'] for x in states)
                        checks.append(dict(step=step,exact=exact,logits=error,states=states));report['checks']=checks;save()
                        if not exact:
                            torch.save(dict(actual=actual,expected=expected,actual_state=actual_state,expected_state=expected_state),out/f'counterexample-cpu-gpu-q4-step{step}.pth')
                            raise AssertionError('Canonical CPU/GPU Q4 decode must be EXACT; no fallback gate relaxation')
                        del p,q,actual,expected,actual_state,expected_state
                del cpu_request,gpu_request
                report['q4_generation_diagnostic']=asdict(generate_flash_next_text(core,reader,tokenizer,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16))
                report['expert_stats']=stats();report['completed']=True;save();return
            q=json.loads(Path(a.qualification).read_text())
            if not q.get('completed') or len(q.get('checks',[]))!=3 or not all(x['exact'] for x in q['checks']) or (
                    q['libraries']!=libraries or q['quality_proof_sha256']!=report['quality_proof_sha256']):
                raise ValueError('Exact independent canonical Q4 full-state proof required')
            report['qualification_sha256']=hashlib.sha256(Path(a.qualification).read_bytes()).hexdigest()
            if a.stage=='profile':
                activate('q4-transient-gemm')
                generate_flash_next_text(core,reader,tokenizer,PROMPT,max_new_tokens=4,eos_token_ids=(),prefill_chunk_size=16)
                with torch.cuda.nvtx.range('venturi.q4.canonical-full'):
                    result=generate_flash_next_text(core,reader,tokenizer,PROMPT,max_new_tokens=4,eos_token_ids=(),prefill_chunk_size=16)
                report.update(completed=True,profile_result=asdict(result),expert_stats=stats());save();return
            modes=('fp8-gpu-lru','q4-transient-gemm')
            for repeat in range(3):
                for mode in (modes if repeat%2==0 else tuple(reversed(modes))):
                    start=time.perf_counter();plan=activate(mode)
                    generate_flash_next_text(core,reader,tokenizer,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
                    setup_seconds=time.perf_counter()-start;converted=sum(x.converted for x in archives)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=process_memory();epoch=time.time()
                    result=generate_flash_next_text(core,reader,tokenizer,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
                    finish=time.time();after=process_memory()
                    if converted!=sum(x.converted for x in archives):raise AssertionError('Unexpected scored Q4 conversion, not steady state')
                    expected=quality['fp8_generation_diagnostic']['generated_ids'] if mode==modes[0] else q['q4_generation_diagnostic']['generated_ids']
                    if list(result.generated_ids)!=expected:raise AssertionError('Each precision must match its own qualified diagnostic IDs')
                    row=dict(mode=mode,repeat=repeat,plan=plan,setup_and_32token_warmup_seconds=setup_seconds,result=asdict(result),
                        scored_started_epoch=epoch,scored_finished_epoch=finish,decode_tps=result.decode_tokens/result.decode_seconds,
                        request_tps=len(result.generated_ids)/result.total_seconds,memory_before=before,memory_after=after,
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),expert_stats=stats())
                    report['records'].append(row);save();print(json.dumps(dict(mode=mode,repeat=repeat,decode_tps=row['decode_tps'])),flush=True)
            report['summary']=[dict(mode=mode,repeats=3,decode_tps=sum(r['result']['decode_tokens'] for r in report['records'] if r['mode']==mode)/sum(r['result']['decode_seconds'] for r in report['records'] if r['mode']==mode),request_tps=sum(len(r['result']['generated_ids']) for r in report['records'] if r['mode']==mode)/sum(r['result']['total_seconds'] for r in report['records'] if r['mode']==mode)) for mode in modes]
            report.update(completed=True,throughput_scored=True);save()
    finally:
        report['memory_at_exit']=process_memory();report['archives_at_exit']=[x.stats for x in archives];save();clear()


if __name__=='__main__':main()
