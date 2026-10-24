"""Greedy generation over explicit Venturi model sessions, with local tokenization."""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from time import perf_counter

import torch

from .models.gpt_neox import GPTNeoXModel, validate_token_ids


class LocalTokenizer:
    """Load only tokenizer.json, with no network access or model Python execution."""

    def __init__(self, path: str | Path, *, vocab_size: int):
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install Venturi's generation extra: pip install -e '.[generate]'"
            ) from error
        self.path = Path(path).expanduser().resolve(strict=True)
        content = self.path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        try:
            self._tokenizer = Tokenizer.from_str(content.decode("utf-8"))
        except Exception as error:
            raise ValueError(f"Invalid tokenizer.json: {error}") from error
        # Disable serialized serving policy: generation supplies one raw prompt,
        # without truncation, padding, or postprocessor-inserted special tokens.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._ids = frozenset(
            self._tokenizer.get_vocab(with_added_tokens=True).values()
        )
        if not self._ids or min(self._ids) < 0 or max(self._ids) >= vocab_size:
            raise ValueError(
                "Tokenizer vocabulary does not fit the model embedding vocabulary"
            )
        self.vocab_size = vocab_size

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise ValueError("prompt must be a string")
        try:
            return tuple(self._tokenizer.encode(text, add_special_tokens=False).ids)
        except Exception as error:
            raise ValueError(f"Tokenizer encoding failed: {error}") from error

    def decode(self, ids) -> str:
        if any(token not in self._ids for token in ids):
            raise ValueError("Model emitted a token ID absent from tokenizer.json")
        try:
            return self._tokenizer.decode(list(ids), skip_special_tokens=True)
        except Exception as error:
            raise ValueError(f"Tokenizer decoding failed: {error}") from error


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
    model: GPTNeoXModel,
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
        if session.model is not model or session.capacity < len(prompt_ids) + max_new_tokens:
            raise ValueError("Provided session must belong to this model and fit the request")
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
    model: GPTNeoXModel, tokenizer: LocalTokenizer, prompt: str, **kwargs
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
