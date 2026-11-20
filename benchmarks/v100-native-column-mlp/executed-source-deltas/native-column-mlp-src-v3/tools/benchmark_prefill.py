"""Warm, reused-session FP16 prefill benchmark; never downloads model weights."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

from venturi import load_checkpoint
from venturi.generation import generate_tokens
from venturi.models import GPTNeoXModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 128, 512, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--execution", choices=("eager", "cuda-graph"), default="eager")
    parser.add_argument("--reference", help="Compare logits with another run's trusted logits.pt")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or any(n < 1 for n in args.lengths):
        parser.error("lengths, warmup and repeats must be positive")
    torch.set_num_threads(8)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    model = GPTNeoXModel(load_checkpoint(args.checkpoint, require_float_dtype="F16"))
    if max(args.lengths) + 1 > model.config.max_position_embeddings:
        parser.error("prompt plus output exceeds the model context")
    reference = torch.load(args.reference, weights_only=True) if args.reference else None
    saved, results = {}, []
    for length in args.lengths:
        # Fixed, nontrivial IDs; identical on every process/run, no tokenizer cost.
        ids = tuple((i * 7919 + 17) % model.config.vocab_size for i in range(length))
        session = (
            model.create_cuda_graph_session(
                length + 1, decode_start=length, decode_steps=1,
                prefill_lengths=(length,),
            ) if args.execution == "cuda-graph" else model.create_session(length + 1)
        )
        for _ in range(args.warmup):
            generate_tokens(model, ids, max_new_tokens=1, eos_token_ids=(), session=session)
        torch.cuda.reset_peak_memory_stats(model.device)
        baseline = torch.cuda.memory_allocated(model.device)
        runs = [
            generate_tokens(model, ids, max_new_tokens=1, eos_token_ids=(), session=session)
            for _ in range(args.repeats)
        ]
        peak = torch.cuda.max_memory_allocated(model.device)
        # Untimed checks also exercise reset, changed input, and the populated KV.
        parity = []
        for label, prompt in (("original", ids), ("changed", tuple(reversed(ids))), ("reset", ids)):
            session.reset()
            logits = session.forward(prompt).clone().cpu()
            next_logits = session.forward([23]).clone().cpu()
            for name, tensor in (("prefill", logits), ("cached_next", next_logits)):
                key = f"{length}/{label}/{name}"
                saved[key] = tensor
                if reference is not None:
                    expected = reference[key]
                    parity.append({
                        "key": key,
                        "byte_equal": torch.equal(tensor.view(torch.int16), expected.view(torch.int16)),
                        "max_abs_error": float((tensor.float() - expected.float()).abs().max()),
                        "same_argmax": int(tensor.argmax()) == int(expected.argmax()),
                    })
        seconds = [run.prefill_seconds for run in runs]
        result = {
            "prompt_tokens": length, "runs_seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "prompt_tokens_per_second": length * len(seconds) / sum(seconds),
            "preparation_seconds": getattr(session, "preparation_seconds", 0.0),
            "allocator_baseline_bytes": baseline, "allocator_peak_bytes": peak,
            "identical_first_tokens": len({run.generated_ids for run in runs}) == 1,
            "reference_checks": parity,
        }
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("runs_seconds", "reference_checks")}), flush=True)
        del session
    torch.save(saved, out / "logits.pt")
    if args.profile:
        session = model.create_session(max(args.lengths) + 1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            generate_tokens(model, ids, max_new_tokens=1, session=session)
        (out / "cpu.txt").write_text(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
        operators = [{"name": e.key, "calls": e.count} for e in profile.key_averages()]
    else:
        operators = None
    root = Path(__file__).resolve().parents[1]
    report = {
        "execution": args.execution, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(model.device),
        "warmup": args.warmup, "repeats": args.repeats, "results": results,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src").rglob("*.py"))},
        "operators": operators,
        "scope": "Single sequence, reused fixed-capacity session; synchronized prefill plus finite-checked greedy first-token selection. Loading, tokenization, graph preparation and subsequent decode excluded. Allocator is PyTorch-only, not device/process residency. Synthetic IDs, not a language-quality benchmark. Profile is an extra untimed eager run; CPU-only operator evidence, not CUDA kernel timing.",
    }
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    if reference is not None and not all(c["byte_equal"] for r in results for c in r["reference_checks"]):
        raise SystemExit("Reference byte parity failed; inspect benchmark.json")


if __name__ == "__main__":
    main()
