"""Experimental FP8-derived Q4: independent execution and SHORT quality screens.

Not BF16 quality, AWQ/GPTQ, full native engine, CPU expert compute, TP/PP/MTP,
long context or serving. Lazy conversion/setup excluded only when explicitly
reported; no complete Q4 checkpoint artifact implied. No gate relaxation.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

from benchmark_flash_next_full import checked_payloads,process_memory
from flash_next_hybrid_metrics import difference,state_to_cpu,compare_state
from venturi.checkpoint import inspect_checkpoint
from venturi.generation import LocalTokenizer
from venturi.flash_next_generation import generate_flash_next_text
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_moe import MoEGeometry
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_q4 import LazyQ4ExpertArchive,PackedQ4Dispatch
from venturi.models.flash_next_request import FlashNextRequest
from venturi.ops.flash_next_q4 import NativeQ4

PROMPT='<|im_start|>user\n用一句话介绍你自己。<|im_end|>\n<|im_start|>assistant\n'
MODES=('fp8-gpu-lru','q4-packed')


def write(path,value):
    temp=path.with_suffix('.json.part');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def quality_delta(reference,quantized):
    p,q=F.log_softmax(reference.float(),-1),F.log_softmax(quantized.float(),-1)
    return dict(kl_fp8_to_q4=float((p.exp()*(p-q)).sum()),
                teacher_top1_same=int(p.argmax())==int(q.argmax()),
                fp8_top1=int(p.argmax()),q4_top1=int(q.argmax()),
                fp8_top1_logprob_delta=float(q[p.argmax()]-p[p.argmax()]),
                max_logit_difference=float((reference.float()-quantized.float()).abs().max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','fp8-library','q4-library','output','archive-root']:p.add_argument('--'+name,required=True)
    p.add_argument('--stage',choices=['correctness','benchmark'],required=True)
    p.add_argument('--qualification')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    directory=Path(a.checkpoint);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');assert torch.cuda.get_device_capability(device)==(7,0)
    torch.cuda.set_per_process_memory_fraction(.85)
    proof,manifest=checked_payloads(directory);cp=inspect_checkpoint(directory);g=MoEGeometry.from_dict(dict(cp.config))
    token=LocalTokenizer(directory/'tokenizer.json',vocab_size=cp.config['text_config']['vocab_size'])
    start=time.perf_counter()
    loaded=load_weight_decoded_fp8_text(cp,max_core_bytes=12*2**30,max_expert_bytes=64*2**20,max_sequence=256,
        attention_backend='sdpa',execution_device='cuda:0',execution_dtype=torch.float16,max_execution_core_bytes=12*2**30)
    core=loaded.core;torch.cuda.synchronize()
    native=NativeQ4(a.q4_library)
    libraries={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('fp8',a.fp8_library),('q4',a.q4_library)]}
    report=dict(scope=__doc__,stage=a.stage,source_revision=manifest['hf_revision'],manifest_sha256=proof['manifest_sha256'],
        source='FP8 decoded FP16 then RTN Q4; NOT original BF16-derived Q4',libraries=libraries,
        prompt=PROMPT,prompt_ids=token.encode(PROMPT),output_tokens=32,decode_transitions=31,prefill_chunk=16,
        context_reservation=256,concurrency=1,loading_seconds=time.perf_counter()-start,
        gpu_retained_budget_per_layer=256*2**20,cpu_retained_decoded_bytes=0,records=[],
        native_full_model=False,complete_q4_checkpoint=False,throughput_scored=False,completed=False)
    def save():write(out/'report.json',report)
    save()
    if a.stage=='benchmark':
        q=json.loads(Path(a.qualification).read_text())
        if not q.get('completed') or q['manifest_sha256']!=proof['manifest_sha256'] or q['libraries']!=libraries:
            raise ValueError('Completed matching independent execution AND quantization screens required')
        report['qualification_sha256']=hashlib.sha256(Path(a.qualification).read_bytes()).hexdigest()
    archive_root=Path(a.archive_root)
    archives=[]
    for i in range(len(core.layers)):
        source=BlockFP8ExpertStore(cp,g,f'model.language_model.layers.{i}.mlp.experts',decode_strategy='vectorized')
        folder=archive_root/f'layer-{i:02d}'
        archives.append(LazyQ4ExpertArchive(source,folder,source_identity=dict(revision=manifest['hf_revision'],
            manifest_sha256=proof['manifest_sha256'],layer=i,decode_dtype='FP16'),create=not folder.exists()))
    dispatches=[]
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
        if mode==MODES[0]:
            return bind_fp8_expert_policy(core,cp,backend='cuda-byte-decode',retained_bytes_per_layer=(256*2**20,)*len(core.layers),library=a.fp8_library)
        for layer,archive in zip(core.layers,archives):
            d=PackedQ4Dispatch(archive,native,device=device,max_resident_bytes=256*2**20)
            dispatches.append(d);layer.mlp.expert_store=archive;layer.mlp.expert_dispatch=d
        return dict(mode=mode,packed_resident_budget=256*2**20*len(core.layers),decoded_retained_bytes=0)
    def selected(mode):
        for layer,d in zip(core.layers,dispatches):
            layer.mlp.expert_dispatch=d if mode=='packed' else d.decoded_reference
    def stats():
        return [d.stats for d in dispatches] if dispatches else [layer.mlp.expert_store.stats for layer in core.layers]
    try:
        with PreparedNGramTable(next(iter(loaded.tables.values()))) as reader:
            if a.stage=='correctness':
                activate(MODES[0]);teacher=[]
                def capture(module,args,result):
                    teacher.append(dict(input_ids=args[0][0].tolist(),last_logits=result[0][0,-1].detach().cpu()))
                handle=core.register_forward_hook(capture)
                try:
                    baseline=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
                finally:handle.remove()
                assert len(teacher)==32
                report['fp8_generation_diagnostic']=asdict(baseline)
                report['teacher_input_ids']=[r['input_ids'] for r in teacher]
                save();activate(MODES[1])
                ref_request,packed_request=FlashNextRequest(core,reader),FlashNextRequest(core,reader)
                execution_checks=[];quality_checks=[];counterexample_saved=False
                with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                    for step,row in enumerate(teacher):
                        selected('decoded');ref=ref_request.preview(row['input_ids']);ref_request.commit(ref)
                        ref_logits=ref.logits.cpu();ref_state=state_to_cpu(ref_request.state)
                        selected('packed');packed=packed_request.preview(row['input_ids']);packed_request.commit(packed)
                        got=packed.logits.cpu();got_state=state_to_cpu(packed_request.state)
                        error=difference(got,ref_logits,kind='logits');states=compare_state(got_state,ref_state)
                        passed=error['passed'] and all(x['passed'] for x in states) and torch.equal(got.argmax(-1),ref_logits.argmax(-1))
                        execution_checks.append(dict(step=step,passed=passed,logits=error,states=states))
                        quality_checks.append(dict(step=step,**quality_delta(row['last_logits'],ref_logits[0,-1])))
                        report.update(execution_checks=execution_checks,quality_checks=quality_checks,archive=[a.stats for a in archives]);save()
                        if not passed and not counterexample_saved:
                            torch.save(dict(step=step,input_ids=row['input_ids'],packed_logits=got,decoded_logits=ref_logits,
                                packed_state=got_state,decoded_state=ref_state),out/'counterexample-packed-vs-q4-decoded.pth')
                            counterexample_saved=True
                        del ref,packed,got,ref_logits,ref_state,got_state
                report['execution_passed']=all(x['passed'] for x in execution_checks)
                kl=[x['kl_fp8_to_q4'] for x in quality_checks]
                report['quality_summary']=dict(mean_kl=sum(kl)/len(kl),max_kl=max(kl),teacher_top1_agreement=sum(x['teacher_top1_same'] for x in quality_checks)/len(quality_checks),
                    scope='32 final-position distributions on ONE FP8-greedy teacher trace; NOT held-out task quality/perplexity')
                s=report['quality_summary'];report['quantization_screen_passed']=s['mean_kl']<=.1 and s['max_kl']<=.3 and s['teacher_top1_agreement']>=.8
                del teacher,ref_request,packed_request
                selected('packed')
                diagnostic=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
                report['q4_generation_diagnostic']=asdict(diagnostic)
                report['generated_ids_same_as_fp8']=diagnostic.generated_ids==baseline.generated_ids
                report['expert_stats']=stats()
                report['completed']=report['execution_passed'] and report['quantization_screen_passed']
                save()
                if not report['completed']:
                    raise AssertionError('Q4 screen failed; diagnostic text is not proof of equivalence or quality; no TPS stage')
                return
            for repeat in range(3):
                for mode in (MODES if repeat%2==0 else tuple(reversed(MODES))):
                    setup_start=time.perf_counter();plan=activate(mode)
                    warmup=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
                    setup_seconds=time.perf_counter()-setup_start
                    converted_before=sum(x.converted for x in archives)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=process_memory();epoch=time.time()
                    result=generate_flash_next_text(core,reader,token,PROMPT,max_new_tokens=32,eos_token_ids=(),prefill_chunk_size=16)
                    finish=time.time();after=process_memory()
                    if converted_before!=sum(x.converted for x in archives):
                        raise AssertionError('Unexpected scored RTN conversion; retain partial data, not steady-state TPS')
                    expected=q['fp8_generation_diagnostic' if mode==MODES[0] else 'q4_generation_diagnostic']['generated_ids']
                    if list(result.generated_ids)!=expected:raise AssertionError('Scored greedy output differs from its own qualified precision path')
                    record=dict(mode=mode,repeat=repeat,plan=plan,setup_and_32token_warmup_seconds=setup_seconds,
                        scored_started_epoch=epoch,scored_finished_epoch=finish,result=asdict(result),
                        decode_tps=result.decode_tokens/result.decode_seconds,request_tps=32/result.total_seconds,
                        memory_before=before,memory_after=after,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),expert_stats=stats())
                    report['records'].append(record);save();print(json.dumps(dict(mode=mode,repeat=repeat,decode_tps=record['decode_tps'])),flush=True)
            report['summary']=[dict(mode=mode,repeats=3,
                decode_tps=sum(r['result']['decode_tokens'] for r in report['records'] if r['mode']==mode)/sum(r['result']['decode_seconds'] for r in report['records'] if r['mode']==mode),
                request_tps=sum(len(r['result']['generated_ids']) for r in report['records'] if r['mode']==mode)/sum(r['result']['total_seconds'] for r in report['records'] if r['mode']==mode)) for mode in MODES]
            report.update(throughput_scored=True,completed=True);save()
    finally:
        report['memory_at_exit']=process_memory();report['archives_at_exit']=[x.stats for x in archives];save();clear()


if __name__=='__main__':main()
