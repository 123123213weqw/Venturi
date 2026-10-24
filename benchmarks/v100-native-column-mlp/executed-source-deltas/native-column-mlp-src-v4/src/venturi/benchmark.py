"""Repeated single-request dense generation, without model downloads or conversion."""

from dataclasses import asdict
from statistics import median

import torch

from .generation import _synchronize, generate_tokens


def _rate(tokens, seconds):
    return tokens / seconds if tokens and seconds > 0 else None


def _distribution(values):
    return {"min": min(values), "median": median(values), "max": max(values)}


def benchmark_tokens(
    model,
    prompt_ids,
    *,
    max_new_tokens=32,
    warmup=1,
    repeats=5,
    ignore_eos=False,
    session=None,
):
    """Measure an already loaded model; create an independent session each run.

    CUDA memory counters are process/device-wide PyTorch allocator counters, not
    total VRAM usage. Run exclusively on the selected device: resetting peaks
    is not compatible with concurrent allocator profiling in this process.
    Tokenization, text decoding, loading, and process startup are not timed.
    """
    for name, value, minimum in (
        ("warmup", warmup, 0),
        ("repeats", repeats, 1),
        ("max_new_tokens", max_new_tokens, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(ignore_eos) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    options = {"max_new_tokens": max_new_tokens}
    if session is not None:
        options["session"] = session
    if ignore_eos:
        options["eos_token_ids"] = ()
    for _ in range(warmup):
        generate_tokens(model, prompt_ids, **options)
    records = []
    for _ in range(repeats):
        memory = None
        if model.device.type == "cuda":
            _synchronize(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
            memory = {
                "baseline_allocated_bytes": torch.cuda.memory_allocated(model.device),
                "baseline_reserved_bytes": torch.cuda.memory_reserved(model.device),
            }
        result = generate_tokens(model, prompt_ids, **options)
        if memory is not None:
            memory.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device),
            )
            memory["extra_peak_allocated_bytes"] = (
                memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"]
            )
        count = len(result.generated_ids)
        records.append({
            **asdict(result),
            "output_tokens": count,
            # Prefill emits the first token; only subsequent tokens are decode.
            "decode_tokens": max(0, count - 1),
            "prefill_tokens_per_second": _rate(
                len(result.prompt_ids), result.prefill_seconds
            ),
            "decode_tokens_per_second": _rate(max(0, count - 1), result.decode_seconds),
            "request_tokens_per_second": _rate(count, result.total_seconds),
            "cuda_allocator": memory,
        })
    return {
        "schema_version": 1,
        "warmup_requests": warmup,
        "session_policy": "prepared_reused" if session is not None else "fresh_per_request",
        "preparation_seconds": getattr(session, "preparation_seconds", 0.0),
        "measured_requests": repeats,
        "eos_policy": "ignored" if ignore_eos else "model_config",
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": len(records[0]["prompt_ids"]),
        "identical_generated_ids": all(
            run["generated_ids"] == records[0]["generated_ids"] for run in records
        ),
        "time_to_first_token_seconds": _distribution([
            run["time_to_first_token_seconds"] for run in records
        ]),
        "request_seconds": _distribution([run["total_seconds"] for run in records]),
        "decode_tokens_per_second": _rate(
            sum(run["decode_tokens"] for run in records),
            sum(run["decode_seconds"] for run in records),
        ),
        "request_tokens_per_second": _rate(
            sum(run["output_tokens"] for run in records),
            sum(run["total_seconds"] for run in records),
        ),
        "peak_cuda_allocated_bytes": max(
            run["cuda_allocator"]["peak_allocated_bytes"] for run in records
        ) if model.device.type == "cuda" else None,
        "scope": (
            "Single-request greedy generation; see session_policy for KV lifetime; "
            "TTFT includes validation/session allocation/fences/prefill/first selection, "
            "not loading/tokenization/text decoding/process startup or prior graph preparation. Decode rate excludes "
            "the first token. CUDA memory is process/device-wide PyTorch allocator only; "
            "excludes native allocations and CUDA context. No cross-engine speedup claim."
        ),
        "runs": records,
    }
