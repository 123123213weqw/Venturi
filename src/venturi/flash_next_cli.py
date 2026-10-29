"""Flash-Next intake and opt-in floating reference generation, not native serving."""

import argparse
from dataclasses import asdict
import json
from time import perf_counter

from .checkpoint import inspect_checkpoint
from .flash_next import FlashNextConfig, FlashNextManifest


def _positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('A positive integer is required')
    return n


def _nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError('A nonnegative integer is required')
    return n


def main(argv=None):
    p = argparse.ArgumentParser(prog='venturi flash-next', description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    i = commands.add_parser('inspect', help='Metadata/storage inspection; no generation')
    i.add_argument('path')
    i.add_argument('--headers-dir', help='Explicit header-only intake, not a loaded checkpoint')
    i.add_argument('--profile', choices=('source', 'expert-q8-g32', 'expert-q4-g64'), default='source')
    i.add_argument('--gpu-budget-gib', type=int, nargs='*', default=[])
    i.add_argument('--ram-budget-gib', type=int, default=0)
    i.add_argument('--batch-size', type=int, default=1)
    i.add_argument('--context', type=int, default=256)
    i.add_argument('--include-mtp', action='store_true')
    i.add_argument('--include-vision', action='store_true')
    g = commands.add_parser('generate-reference', help='Experimental single-request raw text; NOT a native/production route')
    g.add_argument('path', help='Complete local checkpoint; no automatic downloads or remote code')
    g.add_argument('--weight-encoding', required=True, choices=('floating', 'fp8-weight-only'))
    g.add_argument('--prompt', help='Raw completion prompt; no automatic chat template/BOS')
    g.add_argument('--tokenizer', help='Local tokenizer.json, defaults to checkpoint directory')
    g.add_argument('--device', default='cpu', help='cpu or explicit cuda:N; no fallback/TP/PP')
    g.add_argument('--dtype', required=True, choices=('float32', 'float16', 'bfloat16'))
    g.add_argument('--core-budget-mib', required=True, type=_positive, help='Source static parameter budget, NOT total RAM/VRAM')
    g.add_argument('--execution-core-budget-mib', type=_positive, help='Converted static parameter budget, defaults to source budget')
    g.add_argument('--expert-budget-mib', type=_positive, default=64)
    g.add_argument('--expert-cache-mib-per-layer', type=_nonnegative, default=0, help='Decoded retained weights only, NOT total memory')
    g.add_argument('--expert-backend', choices=('cpu-tiled', 'cpu-vectorized', 'cuda-byte-decode'))
    g.add_argument('--decoder-library', help='Explicit prebuilt SM70 library; never auto-compiled')
    g.add_argument('--max-sequence', type=_positive, default=256)
    g.add_argument('--max-new-tokens', type=_nonnegative, default=32)
    g.add_argument('--prefill-chunk-size', type=_positive, default=16)
    eos = g.add_mutually_exclusive_group()
    eos.add_argument('--eos-token-id', type=_nonnegative, action='append', dest='eos_token_ids')
    eos.add_argument('--ignore-eos', action='store_true')
    g.add_argument('--plan-only', action='store_true', help='Metadata-only plan, no tokenizer/payload/CUDA initialization')
    a = p.parse_args(argv)
    try:
        if a.command == 'inspect':
            m = (FlashNextManifest.from_headers(a.path, a.headers_dir) if a.headers_dir else
                 FlashNextManifest.from_checkpoint(inspect_checkpoint(a.path)))
            report = m.plan(profile=a.profile, gpu_budgets=[x*2**30 for x in a.gpu_budget_gib],
                ram_budget=a.ram_budget_gib*2**30, include_mtp=a.include_mtp,
                include_vision=a.include_vision, batch_size=a.batch_size, context=a.context)
        else:
            report = _generate(a)
    except (ValueError, OSError, RuntimeError) as error:
        p.exit(1, f'venturi flash-next: {error}\n')
    print(json.dumps(report, indent=2))
    return 0


def _generate(a):
    import torch

    from .flash_next_generation import generate_flash_next_text, resolve_flash_next_eos
    from .text import LocalTokenizer
    from .models.flash_next_expert_cache import CachedExpertStore
    from .models.flash_next_expert_policy import bind_fp8_expert_policy
    from .models.flash_next_loader import (
        load_floating_text, load_weight_decoded_fp8_text,
        plan_floating_text, plan_weight_decoded_fp8_text,
    )
    from .models.flash_next_ngram import PreparedNGramTable

    fp8 = a.weight_encoding == 'fp8-weight-only'
    if not fp8 and (a.expert_backend is not None or a.decoder_library is not None):
        raise ValueError('Expert decoder backend/library requires explicit FP8-weight-only encoding')
    backend = a.expert_backend or 'cpu-tiled'
    if backend == 'cuda-byte-decode':
        if a.decoder_library is None or not a.device.startswith('cuda:') or a.dtype == 'bfloat16':
            raise ValueError('Native decoder requires explicit cuda:N, FP16/FP32 and prebuilt library')
    elif a.decoder_library is not None:
        raise ValueError('Only cuda-byte-decode accepts a decoder library')
    cp = inspect_checkpoint(a.path)
    config = FlashNextConfig.from_dict(dict(cp.config))
    eos_ids, eos_source = resolve_flash_next_eos(cp, override=a.eos_token_ids, ignore=a.ignore_eos)
    options = dict(max_core_bytes=a.core_budget_mib*2**20, max_expert_bytes=a.expert_budget_mib*2**20,
        max_sequence=a.max_sequence, attention_backend='sdpa', execution_device=a.device,
        execution_dtype=getattr(torch, a.dtype),
        max_execution_core_bytes=(a.execution_core_budget_mib or a.core_budget_mib)*2**20)
    plan = (plan_weight_decoded_fp8_text if fp8 else plan_floating_text)(cp, **options)
    plan.update(command='generate-reference', weight_encoding_selection=a.weight_encoding,
        reference_generation_entrypoint=True, native_generation_supported=False,
        whole_official_model_qualified=False, raw_completion_only=True, prefill_chunk_size=a.prefill_chunk_size,
        expert_backend=backend if fp8 else 'floating-file-backed',
        expert_cache_retained_budget=config.layers*a.expert_cache_mib_per_layer*2**20,
        eos_token_ids=eos_ids, eos_source=eos_source, sampling_policy='greedy; generation_config sampling fields NOT applied',
        timing_scope='Request only; excludes loading, tokenization, reader preparation; includes expert/PLE I/O and commits')
    if a.plan_only:
        return dict(plan_only=True, loaded=False, plan=plan)
    if a.prompt is None:
        raise ValueError('--prompt is required unless --plan-only is selected')
    tokenizer = LocalTokenizer(a.tokenizer or cp.directory/'tokenizer.json', vocab_size=config.vocab_size)
    prompt_ids = tokenizer.encode(a.prompt)
    if not prompt_ids or len(prompt_ids)+a.max_new_tokens > a.max_sequence:
        raise ValueError('Nonempty tokenized prompt plus output must fit --max-sequence')
    chunk = min(len(prompt_ids), a.prefill_chunk_size)
    if a.max_new_tokens and (chunk*config.vocab_size > 4*2**20 or chunk*len(prompt_ids) > 4*2**20):
        raise ValueError('Prefill logits/mask budget exceeded; explicitly reduce chunk size')
    started = perf_counter()
    loaded = (load_weight_decoded_fp8_text if fp8 else load_floating_text)(cp, **options)
    budgets = [a.expert_cache_mib_per_layer*2**20]*config.layers
    if fp8:
        policy = bind_fp8_expert_policy(loaded.core, cp, backend=backend, retained_bytes_per_layer=budgets,
            library=a.decoder_library, max_decoded_expert_bytes=a.expert_budget_mib*2**20)
    else:
        for layer, budget in zip(loaded.core.layers, budgets):
            if budget:
                layer.mlp.expert_store = CachedExpertStore(layer.mlp.expert_store, max_resident_bytes=budget)
        policy = dict(backend='floating-file-backed', retained_bytes_per_layer=budgets)
    loading_seconds = perf_counter()-started
    if len(loaded.tables) != 1:
        raise ValueError('Reference request currently requires exactly one PLE layer')
    started = perf_counter()
    with PreparedNGramTable(next(iter(loaded.tables.values()))) as table:
        preparation_seconds = perf_counter()-started
        result = generate_flash_next_text(loaded.core, table, tokenizer, a.prompt,
            max_new_tokens=a.max_new_tokens, prefill_chunk_size=a.prefill_chunk_size,
            eos_token_ids=eos_ids)
    return dict(plan_only=False, loaded=True, plan=plan, expert_policy=policy,
        loading_seconds=loading_seconds, reader_preparation_seconds=preparation_seconds,
        tokenizer=dict(file=tokenizer.path.name, sha256=tokenizer.sha256, chat_template_applied=False),
        result=asdict(result))
