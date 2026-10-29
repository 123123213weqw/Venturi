"""Experimental single-request greedy TEXT reference, not the native engine.

No downloads, model Python, chat template, sampling, MTP, graphs or implicit
placement. Each invocation owns a fresh request. Chunk size is explicit because
chunk/recurrent GDN rounding need not be identical across prefill schedules.
"""

from dataclasses import dataclass, replace
from time import perf_counter, process_time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .text import LocalTokenizer
from .checkpoint import _read_json
from .models.flash_next_request import FlashNextRequest


@dataclass(frozen=True)
class FlashNextGenerationResult:
    prompt_ids: tuple
    generated_ids: tuple
    finish_reason: str
    prefill_chunk_size: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    prefill_cpu_seconds: float
    decode_cpu_seconds: float
    total_cpu_seconds: float
    time_to_first_token_seconds: float | None
    decode_tokens: int  # Excludes the first token selected from prefill logits.
    committed_tokens: int  # Final emitted token is NOT forwarded just to return it.
    text: str | None = None


def resolve_flash_next_eos(checkpoint, *, override=None, ignore=False):
    """Read only stop-token metadata, NEVER execute templates/apply sampling.

    Generation stop IDs can include turn-end tokens in addition to the model's
    ngram segment EOS. They must NOT replace the ngram table's own EOS setting.
    """
    config = checkpoint.config['text_config']
    if ignore:
        return (), 'ignore-eos'
    if override is not None:
        ids, source = tuple(override), 'explicit'
    else:
        path = checkpoint.directory/'generation_config.json'
        if path.exists():
            if path.stat().st_size > 256*2**10:
                raise ValueError('generation_config.json exceeds bounded metadata size')
            value = _read_json(path).get('eos_token_id')
        else:
            value = None
        if value is None:
            value, source = config['eos_token_id'], 'text_config.eos_token_id'
        else:
            source = 'generation_config.json'
        if type(value) is int:
            ids = (value,)
        elif isinstance(value, list):
            ids = tuple(value)
        else:
            raise ValueError('EOS metadata must be an integer or integer list')
    if not 0 < len(ids) <= 128 or any(type(t) is not int or not 0 <= t < config['vocab_size'] for t in ids):
        raise ValueError('A bounded nonempty list of valid integer EOS IDs is required')
    return tuple(dict.fromkeys(ids)), source


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.current_stream(device).synchronize()


def _select(logits):
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError('Flash-Next produced nonfinite logits; no token was selected')
    return int(logits.argmax().item())


def generate_flash_next_tokens(core, table, prompt_ids, *, max_new_tokens=32,
                               eos_token_ids=None, prefill_chunk_size=16):
    """Fresh private request, bounded chunked prefill followed by cached decode.

    Raw completion only. EOS inside the prompt resets ngram segments according
    to the model, but does not terminate prefill/reset decoder state. An emitted
    EOS terminates generation. Empty prompts and implicit truncation are rejected.
    Timing excludes loading/tokenization/reader preparation; includes row I/O,
    expert fetch/decode, model work, request commit and greedy selection. SDPA
    is explicitly MATH here; no unqualified fused backend is selected.
    """
    started, cpu_started = perf_counter(), process_time()
    ids = tuple(prompt_ids)
    # Construction validates core/table metadata only; it neither reads rows nor
    # allocates model state, including on the zero-output path.
    request = FlashNextRequest(core, table)
    vocab = core.embed_tokens.num_embeddings
    if not ids or any(type(t) is not int or not 0 <= t < vocab for t in ids):
        raise ValueError('Nonempty valid integer prompt token IDs required')
    if type(max_new_tokens) is not int or max_new_tokens < 0 or type(prefill_chunk_size) is not int or prefill_chunk_size <= 0:
        raise ValueError('Nonnegative output length and positive explicit prefill chunk size required')
    eos = (table.table_key[0].eos_token_id,) if eos_token_ids is None else tuple(eos_token_ids)
    if any(type(t) is not int or not 0 <= t < vocab for t in eos):
        raise ValueError('Valid integer EOS token IDs required')
    if len(ids)+max_new_tokens > core.max_sequence:
        raise ValueError('Prompt plus output reservation exceeds context; no implicit truncation')
    chunk = min(prefill_chunk_size, len(ids))
    if max_new_tokens and (chunk > table.max_tokens or chunk*vocab > core.max_logits_elements or
                           chunk*len(ids) > core.max_mask_elements):
        raise ValueError('Prefill schedule exceeds explicit logits/mask budget; select a smaller chunk size')
    if max_new_tokens == 0:
        return FlashNextGenerationResult(ids, (), 'length', prefill_chunk_size, 0., 0.,
            perf_counter()-started, 0., 0., process_time()-cpu_started, None, 0, 0)
    device = core.embed_tokens.weight.device
    if device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
        raise ValueError('Flash-Next reference generation/cache path is eager-only, not CUDA graph capture')
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        _sync(device)
        prefill_start, prefill_cpu_start = perf_counter(), process_time()
        first = None
        for offset in range(0, len(ids), prefill_chunk_size):
            proposal = request.preview(ids[offset:offset+prefill_chunk_size])
            # Validate every chunk BEFORE commit, even when only its last
            # prediction will be used; do not hide a nonfinite intermediate.
            if not bool(torch.isfinite(proposal.logits).all()):
                raise RuntimeError('Flash-Next produced nonfinite logits during prefill')
            if offset+prefill_chunk_size >= len(ids):
                first = _select(proposal.logits[0, -1])
            request.commit(proposal)
        _sync(device)
        first_ready = perf_counter()
        prefill_seconds = first_ready-prefill_start
        prefill_cpu_seconds = process_time()-prefill_cpu_start
        generated = [first]
        decode_start, decode_cpu_start = perf_counter(), process_time()
        while len(generated) < max_new_tokens and generated[-1] not in eos:
            proposal = request.preview([generated[-1]])
            token = _select(proposal.logits[0, -1])
            request.commit(proposal)
            generated.append(token)
        _sync(device)
        decode_seconds, decode_cpu_seconds = perf_counter()-decode_start, process_time()-decode_cpu_start
    return FlashNextGenerationResult(ids, tuple(generated), 'eos' if generated[-1] in eos else 'length',
        prefill_chunk_size, prefill_seconds, decode_seconds, perf_counter()-started,
        prefill_cpu_seconds, decode_cpu_seconds, process_time()-cpu_started, first_ready-started,
        len(generated)-1, len(ids)+len(generated)-1)


def generate_flash_next_text(core, table, tokenizer: LocalTokenizer, prompt, **options):
    """Local raw-prompt tokenization only; no guessed BOS/chat configuration."""
    if tokenizer.vocab_size != core.embed_tokens.num_embeddings:
        raise ValueError('Tokenizer and Flash-Next vocabulary differ')
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError('Empty tokenized prompt; supply an explicit nonempty prompt')
    result = generate_flash_next_tokens(core, table, ids, **options)
    return replace(result, text=tokenizer.decode(result.generated_ids))
