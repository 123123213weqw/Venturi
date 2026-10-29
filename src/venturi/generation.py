"""Greedy generation over explicit Venturi model sessions, with local tokenization."""

from dataclasses import dataclass, replace
from time import perf_counter

import torch

from .text import LocalTokenizer, validate_token_ids
from .text.contracts import GenerationModel


@dataclass(frozen=True)
class GenerationResult:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    finish_reason: str
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    kv_cache_bytes: int
    text: str | None = None
    time_to_first_token_seconds: float | None = None


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def _greedy_token(logits):
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Model produced nonfinite logits; no token was selected")
    return int(torch.argmax(logits).item())


def generate_tokens(
    model: GenerationModel,
    prompt_ids,
    *,
    max_new_tokens: int = 32,
    eos_token_ids: tuple[int, ...] | None = None,
    session=None,
) -> GenerationResult:
    """Single-request greedy continuation, bounded by EOS and model context.

    At least one prompt token is required. Reserve prompt+max_new_tokens cache
    positions conservatively. No sampling, beam search, prompt truncation, or
    sliding-window eviction is implicit. The emitted final token need not be
    forwarded through the model merely to return it.

    A supplied session must belong to this model and fit the full reservation.
    Its logical history is reset for each request; allocation/graph preparation
    performed by the caller is outside these request timing fields.
    """
    started = perf_counter()
    validate_token_ids(prompt_ids, model.config.vocab_size)
    prompt_ids = tuple(prompt_ids)
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if len(prompt_ids) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(
            "Prompt plus requested output exceeds max_position_embeddings; no automatic truncation"
        )
    eos = model.config.eos_token_ids if eos_token_ids is None else eos_token_ids
    validate_token_ids(eos, model.config.vocab_size, allow_empty=True)
    if max_new_tokens == 0:
        return GenerationResult(
            prompt_ids, (), "length", 0.0, 0.0, perf_counter() - started, 0
        )
    if session is None:
        session = model.create_session(len(prompt_ids) + max_new_tokens)
    else:
        if (
            session.model is not model
            or session.capacity < len(prompt_ids) + max_new_tokens
        ):
            raise ValueError(
                "Provided session must belong to this model and fit the request"
            )
        validate_request = getattr(session, "validate_request", None)
        if validate_request is not None:
            validate_request(len(prompt_ids), max_new_tokens)
        session.reset()
    select = getattr(session, "select_token", _greedy_token)
    _synchronize(model.device)
    prefill_start = perf_counter()
    logits = session.forward(prompt_ids)
    first = select(logits)
    first_ready = perf_counter()
    prefill_seconds = first_ready - prefill_start
    generated = [first]
    decode_start = perf_counter()
    with torch.inference_mode():
        while len(generated) < max_new_tokens and generated[-1] not in eos:
            logits = session.forward([generated[-1]])
            generated.append(select(logits))
    _synchronize(model.device)
    decode_seconds = perf_counter() - decode_start
    return GenerationResult(
        prompt_ids,
        tuple(generated),
        "eos" if generated[-1] in eos else "length",
        prefill_seconds,
        decode_seconds,
        perf_counter() - started,
        session.cache_bytes,
        time_to_first_token_seconds=first_ready - started,
    )


def generate_text(
    model: GenerationModel, tokenizer: LocalTokenizer, prompt: str, **kwargs
):
    """Raw text completion; no chat template or implicit instruction formatting."""
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Tokenizer and model vocab_size differ")
    ids = tokenizer.encode(prompt)
    if not ids:
        if model.config.bos_token_id is None:
            raise ValueError("Empty prompt requires a configured BOS token")
        ids = (model.config.bos_token_id,)
    result = generate_tokens(model, ids, **kwargs)
    return replace(result, text=tokenizer.decode(result.generated_ids))
