"""Real complete Flash-Next FP8-WEIGHT-only text reference correctness then speed.

Not official W8A8 activation equivalence, vision/MTP, native full engine, TP/PP
or concurrent serving. Requires a fully hash-verified local checkpoint marker.
Static source weights are read-only. GPU0/FP16/SM70/prebuilt byte decoder explicit.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_next_full_oracle import make_full_oracle, run_oracle_step, assert_full_state
from venturi.checkpoint import inspect_checkpoint
from venturi.flash_next_generation import generate_flash_next_text, resolve_flash_next_eos
from venturi.generation import LocalTokenizer
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_request import FlashNextRequest


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def process_memory():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io = {}
    for line in Path('/proc/self/io').read_text().splitlines():
        key, value = line.split(':')
        io[key] = int(value)
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return dict(process_peak_rss_bytes=usage.ru_maxrss*1024, minor_faults=usage.ru_minflt,
        major_faults=usage.ru_majflt, system_mem_available_bytes=available, io=io)


def cache_stats(core):
    return [dict(layer=i, **layer.mlp.expert_store.stats) for i, layer in enumerate(core.layers)
            if hasattr(layer.mlp.expert_store, 'stats')]


def checked_payloads(directory):
    marker = directory/'.venturi-complete.json'
    proof = json.loads(marker.read_text())
    raw = (directory/'.venturi-download-manifest.json').read_bytes()
    if not proof['complete'] or hashlib.sha256(raw).hexdigest() != proof['manifest_sha256']:
        raise ValueError('Complete authenticated checkpoint proof required')
    manifest = json.loads(raw)
    if manifest['hf_revision'] != '236dfdf285828023ca3bcd3f37366c58a3469b13':
        raise ValueError('Only pinned ORIGINAL official FP8 checkpoint for this campaign')
    mapping = {v['filename']:v for v in proof['files']}
    for item in manifest['files']:
        path = directory/item['filename']
        if not path.is_file() or path.stat().st_size != item['bytes'] or mapping[item['filename']]['sha256'] != item['sha256']:
            raise ValueError('Authenticated payload changed or missing: '+item['filename'])
    return proof, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--upstream-modeling',required=True)
    p.add_argument('--upstream-cache',required=True)
    p.add_argument('--upstream-sdpa',required=True)
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--cache-mib-per-layer',type=int,default=256)
    p.add_argument('--profile-only',action='store_true')
    a=p.parse_args()
    if not 4<=a.max_new_tokens<=64 or not 0<a.cache_mib_per_layer<=256:
        raise ValueError('Bounded output/cache campaign required')
    directory, out = Path(a.checkpoint), Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    device=torch.device('cuda:0')
    props=torch.cuda.get_device_properties(device)
    if (props.major,props.minor)!=(7,0):
        raise ValueError('Pinned SM70 correctness/benchmark only')
    torch.cuda.set_per_process_memory_fraction(.85,device)
    proof,manifest=checked_payloads(directory)
    checkpoint=inspect_checkpoint(directory)
    tokenizer=LocalTokenizer(directory/'tokenizer.json',vocab_size=checkpoint.config['text_config']['vocab_size'])
    eos,eos_source=resolve_flash_next_eos(checkpoint)
    started=time.perf_counter()
    loaded=load_weight_decoded_fp8_text(checkpoint,max_core_bytes=12*2**30,max_expert_bytes=64*2**20,
        max_sequence=256,attention_backend='sdpa',execution_device='cuda:0',execution_dtype=torch.float16,
        max_execution_core_bytes=12*2**30)
    torch.cuda.synchronize()
    loading_seconds=time.perf_counter()-started
    core=loaded.core
    report=dict(scope=__doc__,complete_official_payloads=True,manifest_sha256=proof['manifest_sha256'],
        repo=manifest['repo'],hf_revision=manifest['hf_revision'],torch=str(torch.__version__),
        device=props.name,gpu_uuid=str(props.uuid) if hasattr(props,'uuid') else None,
        checkpoint_dtype='BF16 static + original FP8 experts/PLE',execution_dtype='FP16',
        activation_quantization=False,official_w8a8_quality_equivalence=False,native_full_model=False,
        source_static_parameters_shared_with_oracle=True,source_binding_independently_tested=False,
        loading_seconds=loading_seconds,loading_plan=dict(loaded.report),tokenizer_sha256=tokenizer.sha256,
        eos_token_ids=eos,eos_source=eos_source,model_components='All48 text layers/all512 experts available; no vision/MTP',
        timing_scope='Fresh private request; prefill/decode incl PLE/expert IO, CPU decode/host dispatch, GPU work, commits; loading/tokenization/reader prep excluded',
        precision_scope='FP8 weights decoded into FP16 GEMMs, not FP8 Tensor Cores or dynamic activation quantization',
        records=[])
    write(out/'report.json',report)
    def policy(budget):
        return bind_fp8_expert_policy(core,checkpoint,backend='cuda-byte-decode',
            retained_bytes_per_layer=(budget,)*len(core.layers),library=a.library)
    with PreparedNGramTable(next(iter(loaded.tables.values()))) as reader:
        if not a.profile_only:
            policy(0)
            oracle,head,Cache,flags=make_full_oracle(checkpoint,core,reader,modeling=a.upstream_modeling,
                cache_source=a.upstream_cache,sdpa_source=a.upstream_sdpa)
            request,cache=FlashNextRequest(core,reader),Cache()
            checks=[]
            # Exactly two full-width real-weight steps; reference CPU FP8 decode
            # can be slow. Actual generation uses a separate explicit chat prompt.
            tokens=[248045]
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                for step in range(2):
                    proposal=request.preview(tokens)
                    expected=run_oracle_step(oracle,head,cache,tokens)
                    try:
                        torch.testing.assert_close(proposal.logits,expected,atol=0,rtol=0)
                        request.commit(proposal)
                        assert_full_state(request.state,cache)
                    except AssertionError:
                        torch.save(dict(actual=proposal.logits.cpu(),expected=expected.cpu(),tokens=tokens,
                            proposed_state=request._pending.state if request._pending is not None else request.state,
                            reference_keys=cache.keys,reference_values=cache.values,
                            reference_raw_indexer=cache.raw,
                            reference_linear=[dict(conv=cache.get_conv_state(i),recurrent=cache.get_recurrent_state(i))
                                for i in range(len(core.layers))]),out/f'counterexample-step{step}.pth')
                        raise
                    tokens=[int(expected[0,-1].argmax())]
                    checks.append(dict(step=step,next_token=tokens[0],all_logits_exact=True,all_states_exact=True))
                    report['correctness']=checks
                    write(out/'report.json',report)
            report['reference_flags']=flags
            del oracle,head,Cache,request,cache,proposal,expected
            gc.collect()
            torch.cuda.synchronize()
            write(out/'report.json',report)
        else:
            policy(a.cache_mib_per_layer*2**20)
        prompt='<|im_start|>user\n用一句话介绍你自己。<|im_end|>\n<|im_start|>assistant\n'
        prompt_ids=tokenizer.encode(prompt)
        report['prompt']=prompt
        report['prompt_ids']=prompt_ids
        report['chat_format']='Explicit literal ChatML string; NO remote/Jinja template execution'
        if a.profile_only:
            with torch.cuda.nvtx.range('venturi.flash-next.real-profile-only'):
                result=generate_flash_next_text(core,reader,tokenizer,prompt,max_new_tokens=4,
                    eos_token_ids=(),prefill_chunk_size=16)
            report['profile_result']=asdict(result)
            report['throughput_scored']=False
            write(out/'report.json',report)
            return
        # Qualitative smoke output: generated by the complete actual-weight text
        # model, NOT a benchmark score or a broad quality/perplexity evaluation.
        policy(a.cache_mib_per_layer*2**20)
        quality=generate_flash_next_text(core,reader,tokenizer,prompt,max_new_tokens=a.max_new_tokens,
            eos_token_ids=eos,prefill_chunk_size=16)
        report['real_text_smoke']=asdict(quality)
        report['quality_evaluation']='One text smoke sample; no benchmark/perplexity or BF16/W8A8 parity claim'
        write(out/'report.json',report)
        expected_ids=None
        for repeat in range(3):
            modes=(0,a.cache_mib_per_layer) if repeat%2==0 else (a.cache_mib_per_layer,0)
            for budget_mib in modes:
                configuration=policy(budget_mib*2**20)
                warmup=generate_flash_next_text(core,reader,tokenizer,prompt,max_new_tokens=4,
                    eos_token_ids=(),prefill_chunk_size=16)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                before=process_memory()
                scored_started_epoch=time.time()
                result=generate_flash_next_text(core,reader,tokenizer,prompt,max_new_tokens=a.max_new_tokens,
                    eos_token_ids=(),prefill_chunk_size=16)
                after=process_memory()
                scored_finished_epoch=time.time()
                if expected_ids is None:
                    expected_ids=result.generated_ids
                if result.generated_ids!=expected_ids:
                    raise AssertionError('Decoded cache policy changed actual generated IDs')
                record=dict(repeat=repeat,cache_mib_per_layer=budget_mib,expert_policy=configuration,
                    warmup_output_tokens=len(warmup.generated_ids),max_new_tokens=a.max_new_tokens,ignore_eos=True,
                    context_reservation=256,prefill_chunk=16,concurrency=1,result=asdict(result),
                    scored_started_epoch=scored_started_epoch,scored_finished_epoch=scored_finished_epoch,
                    decode_tps=result.decode_tokens/result.decode_seconds,
                    request_tps=len(result.generated_ids)/result.total_seconds,
                    memory_before=before,memory_after=after,cache=cache_stats(core),
                    cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                report['records'].append(record)
                write(out/'report.json',report)
                print(json.dumps(dict(repeat=repeat,cache_mib_per_layer=budget_mib,decode_tps=record['decode_tps'],
                    request_tps=record['request_tps'],cuda_peak_allocated_bytes=record['cuda_peak_allocated_bytes'])),flush=True)
        report['summary']=[]
        for budget in (0,a.cache_mib_per_layer):
            rows=[r for r in report['records'] if r['cache_mib_per_layer']==budget]
            report['summary'].append(dict(cache_mib_per_layer=budget,repeats=len(rows),
                decode_tps=sum(r['result']['decode_tokens'] for r in rows)/sum(r['result']['decode_seconds'] for r in rows),
                request_tps=sum(len(r['result']['generated_ids']) for r in rows)/sum(r['result']['total_seconds'] for r in rows)))
        report['throughput_scored']=True
        report['generated_ids_same_all_policies']=True
        write(out/'report.json',report)


if __name__=='__main__':
    main()
