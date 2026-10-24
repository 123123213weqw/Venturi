"""Real-checkpoint acceptance against separate FP32 and FP16 references.

Transformers is used only by this validation tool, never by Venturi generation.
All inputs are local files. No model/tokenizer download or dtype change to
Venturi's loaded weights occurs here.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile

import torch

from venturi import load_checkpoint
from venturi.generation import LocalTokenizer, generate_text
from venturi.models import GPTNeoXModel


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(2**20):
            result.update(chunk)
    return result.hexdigest()


def compare(actual, expected, *, require_close=True):
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise RuntimeError("Nonfinite logits in actual or reference execution")
    error = (actual.float() - expected.float()).abs()
    # Raw Pythia logits can exceed 1000: one FP16 rounding interval is then
    # larger than a fixed 0.15 absolute threshold. Use a dtype-aware bound;
    # independently qualify the adapter's FP32 mathematics as well.
    rtol = 4 * torch.finfo(torch.float16).eps if actual.dtype == torch.float16 else 1e-5
    bound = 0.002 + rtol * expected.float().abs()
    record = {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "max_error_over_bound": float((error / bound).max()),
        "within_tolerance": bool((error <= bound).all()),
        "argmax_agreement": float(
            (actual.argmax(-1) == expected.argmax(-1)).float().mean()
        ),
    }
    if require_close and not record["within_tolerance"]:
        raise RuntimeError(f"Logits failed dtype-aware numerical tolerance: {record}")
    return record


def reference_model(cpu, device, dtype=torch.float32, *, attention=None):
    import transformers

    config = transformers.GPTNeoXConfig.from_dict(dict(cpu.config))
    config._attn_implementation = attention or (
        "sdpa" if dtype == torch.float16 else "eager"
    )
    reference = transformers.GPTNeoXForCausalLM(config).eval()
    frequencies = {
        name: module.inv_freq.clone()
        for name, module in reference.named_modules()
        if hasattr(module, "inv_freq")
    }
    reference.to(device=device, dtype=dtype)
    # model.to(float16) also casts nonpersistent RoPE buffers. Restore their
    # original FP32 values to match the declared FP32-frequency contract.
    for name, frequency in frequencies.items():
        reference.get_submodule(name).inv_freq = frequency.to(device=device)
    state = {}
    for key in reference.state_dict():
        source = "embed_out.weight" if key == "lm_head.weight" else key
        if source not in cpu.weights:
            raise ValueError(f"Reference parameter {key} absent in source checkpoint")
        state[key] = cpu.weights[source]
    reference.load_state_dict(state, strict=True)
    return reference


def reference_greedy(reference, ids, budget, eos, device):
    options = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(reference.forward).parameters
        else {}
    )
    output = reference(torch.tensor([ids], device=device), use_cache=True, **options)
    generated = []
    for index in range(budget):
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
        if token in eos or index + 1 == budget:
            break
        output = reference(
            torch.tensor([[token]], device=device),
            use_cache=True,
            past_key_values=output.past_key_values,
            **options,
        )
    return generated


@torch.inference_mode()
def run(path, device, loader, tokenizer_path, max_new_tokens):
    import transformers
    import tokenizers

    loaded = load_checkpoint(
        path, device=device, loader=loader, require_float_dtype="F16"
    )
    model = GPTNeoXModel(loaded)
    tokenizer = LocalTokenizer(
        tokenizer_path or loaded.checkpoint.directory / "tokenizer.json",
        vocab_size=model.config.vocab_size,
    )
    cpu = load_checkpoint(path, device="cpu", loader="safetensors")
    reference = reference_model(cpu, model.device)
    reference_fp16 = reference_model(cpu, model.device, torch.float16)
    validation_fp32 = GPTNeoXModel(
        replace(
            cpu,
            weights={
                name: tensor.to(device=model.device, dtype=torch.float32)
                if tensor.is_floating_point()
                else tensor.to(model.device)
                for name, tensor in cpu.weights.items()
            },
        )
    )
    report = {
        "scope": "FP16 dense GPT-NeoX text generation; no Q4/full-Qwen inference claim",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers_reference": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "cpu_threads": torch.get_num_threads(),
            "device": str(model.device),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "files": {
            file.name: {"bytes": file.stat().st_size, "sha256": digest(file)}
            for file in loaded.checkpoint.files
        },
        "tokenizer": {"file": tokenizer.path.name, "sha256": tokenizer.sha256},
        "load": asdict(loaded.stats),
        "numerical_criteria": {
            "formula": "abs(actual-reference) <= atol + rtol * abs(reference)",
            "atol": 0.002,
            "fp16_rtol": 4 * torch.finfo(torch.float16).eps,
            "fp32_rtol": 1e-5,
            "reference_dtype": "Separate FP32 and FP16 models from the same stored learned parameters",
            "rope": "FP32 inverse frequencies preserved in both references, not rounded by model.to(float16)",
            "notes": "FP16 argmax can differ from FP32 due to large-logit rounding; agreement is recorded, not assumed",
            "qualification": "Short FP32 prefill vs eager; same-schedule cached FP16/FP32 vs SDPA references. Long cached/full-batch drift is diagnostic, not hidden by larger tolerances.",
        },
        "cases": [],
    }
    if model.device.type == "cuda":
        report["runtime"]["gpu"] = torch.cuda.get_device_name(model.device)
    for prompt in ("Once upon a time", "The capital of France is", "def fibonacci(n):"):
        print(f"Validating {prompt!r}", flush=True)
        ids = list(tokenizer.encode(prompt))
        session = model.create_session(len(ids) + max_new_tokens)
        actual = session.forward(ids, all_logits=True)
        expected = reference(
            torch.tensor([ids], device=model.device), use_cache=False
        ).logits[0]
        prefill = compare(actual, expected)
        fp32_math = compare(
            validation_fp32.create_session(len(ids)).forward(ids, all_logits=True),
            expected,
        )
        same_precision = compare(
            actual,
            reference_fp16(
                torch.tensor([ids], device=model.device), use_cache=False
            ).logits[0],
        )
        forced = []
        # Teacher-force the independent reference's greedy tokens; every step
        # compares the same input sequence instead of drifting onto different text.
        for _ in range(max_new_tokens - 1):
            token = int(expected[-1].argmax().item())
            ids.append(token)
            actual = session.forward([token])
            expected = reference(
                torch.tensor([ids], device=model.device), use_cache=False
            ).logits[0]
            forced.append(compare(actual, expected[-1]))
        cached_vs_full = (
            compare(actual, model.create_session(len(ids)).forward(ids))
            if forced
            else None
        )
        generated = generate_text(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        # Independent cached greedy loops, not Venturi's generation scheduler.
        reference_ids = list(tokenizer.encode(prompt))
        reference_generated = reference_greedy(
            reference,
            reference_ids,
            max_new_tokens,
            model.config.eos_token_ids,
            model.device,
        )
        same_precision_generated = reference_greedy(
            reference_fp16,
            reference_ids,
            max_new_tokens,
            model.config.eos_token_ids,
            model.device,
        )
        agreement = list(generated.generated_ids) == reference_generated
        report["cases"].append(
            {
                "prompt": prompt,
                "prefill_vs_fp32": prefill,
                "prefill_vs_same_precision_reference": same_precision,
                "fp32_adapter_vs_fp32_reference": fp32_math,
                "teacher_forced_decode_vs_fp32": forced,
                "cached_vs_uncached_fp16": cached_vs_full,
                "generation": asdict(generated),
                "reference_generated_ids": reference_generated,
                "greedy_ids_match_fp32_reference": agreement,
                "same_precision_reference_generated_ids": same_precision_generated,
                "greedy_ids_match_same_precision_reference": list(
                    generated.generated_ids
                )
                == same_precision_generated,
            }
        )
    # Exercise a nonzero RoPE offset and a cached multi-token chunk.
    long_ids = ([42, 111, 314] * 172)[:512]
    long_session = model.create_session(515)
    long_session.forward(long_ids)
    chunk = [83, 97, 101]
    actual = long_session.forward(chunk, all_logits=True)
    expected = reference(
        torch.tensor([long_ids + chunk], device=model.device), use_cache=False
    ).logits[0, -3:]
    report["cached_chunk_at_position_512"] = compare(actual, expected)
    same_precision = reference_fp16(
        torch.tensor([long_ids + chunk], device=model.device), use_cache=False
    ).logits[0, -3:]
    report["cached_chunk_at_position_512_vs_same_precision"] = compare(
        actual, same_precision, require_close=False
    )
    reference_past = reference_fp16(
        torch.tensor([long_ids], device=model.device), use_cache=True
    ).past_key_values
    reference_cached = reference_fp16(
        torch.tensor([chunk], device=model.device),
        past_key_values=reference_past,
        use_cache=True,
    ).logits[0]
    report["cached_chunk_at_position_512_vs_cached_fp16_reference"] = compare(
        actual, reference_cached
    )
    full_fp16 = model.create_session(515).forward(long_ids + chunk, all_logits=True)[
        -3:
    ]
    report["cached_chunk_at_position_512_vs_uncached_fp16"] = compare(
        actual, full_fp16, require_close=False
    )
    fp32_session = validation_fp32.create_session(515)
    fp32_session.forward(long_ids)
    fp32_actual = fp32_session.forward(chunk, all_logits=True)
    report["cached_chunk_at_position_512_fp32_vs_full_eager"] = compare(
        fp32_actual, expected, require_close=False
    )
    fp32_sdpa = reference_model(cpu, model.device, attention="sdpa")
    past = fp32_sdpa(
        torch.tensor([long_ids], device=model.device), use_cache=True
    ).past_key_values
    fp32_cached = fp32_sdpa(
        torch.tensor([chunk], device=model.device), past_key_values=past, use_cache=True
    ).logits[0]
    report["cached_chunk_at_position_512_fp32_math"] = compare(fp32_actual, fp32_cached)
    report["known_limitation"] = (
        "Cached vs full-batch arithmetic is not numerically invariant on this checkpoint; raw drift and argmax agreement are retained. Same-schedule cached references are qualified independently."
    )
    report["all_greedy_ids_match_same_precision_reference"] = all(
        case["greedy_ids_match_same_precision_reference"] for case in report["cases"]
    )
    report["all_greedy_ids_match_fp32_reference"] = all(
        case["greedy_ids_match_fp32_reference"] for case in report["cases"]
    )
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--loader", choices=("siphon", "safetensors"), default="siphon")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        parser.error("max-new-tokens must be in [1,128]")
    report = run(
        args.path, args.device, args.loader, args.tokenizer, args.max_new_tokens
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as file:
        json.dump(report, file, indent=2)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
