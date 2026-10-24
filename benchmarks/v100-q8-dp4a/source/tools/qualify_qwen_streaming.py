"""Complete Q8 streaming forward/generation diagnostic, NOT an engine benchmark.

The native numerical layers are owned CUDA operators. Python performs explicit
weight/KV staging, table setup and synchronization. Greedy selection can use an
owned native GPU operator (default) or the explicit host CPU reference.
An optional allocating CPU Torch profile follows the same native-generated input
sequence. It is not an independent upstream architecture or BF16 quality study.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import torch

from venturi.models import StreamingQwenModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, TorchBackend, prepare_greedy


def metrics(actual, reference):
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    if a.shape != b.shape or not bool(a.isfinite().all() and b.isfinite().all()):
        raise ValueError("Compared tensors must be shape-matched and finite")
    delta = a - b
    norm = float(b.norm())
    return dict(
        max_abs_error=float(delta.abs().max()) if delta.numel() else 0.0,
        relative_l2=float(delta.norm()) / max(norm, 1e-30),
        elements=a.numel(),
    )


def logits_comparison(actual, reference):
    result = metrics(actual, reference)
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    ac, bc = a - a.mean(), b - b.mean()
    cosine = (
        (1.0 if torch.equal(a, b) else 0.0)
        if not bool(ac.norm() * bc.norm())
        else float((ac @ bc) / (ac.norm() * bc.norm()))
    )
    logp, logq = torch.log_softmax(b, 0), torch.log_softmax(a, 0)
    kl = float((logp.exp() * (logp - logq)).sum())
    result.update(
        centered_cosine=cosine,
        reference_kl_to_native=kl,
        native_top1=int(a.argmax()),
        reference_top1=int(b.argmax()),
    )
    # Fixed initial diagnostic gates: do not widen them after seeing results.
    # Passing a few teacher-forced steps is NOT corpus or original-BF16 quality.
    result["passed"] = bool(
        result["native_top1"] == result["reference_top1"]
        and result["relative_l2"] <= 0.02
        and cosine >= 0.999
        and kl <= 0.002
    )
    return result


def cpu_greedy(logits):
    value = logits.detach().cpu()
    if value.ndim != 1 or not bool(value.isfinite().all()):
        raise ValueError("Cannot sample nonfinite/nonvector logits")
    return int(value.argmax())


def planned_greedy(logits, plan, out, workspace):
    """Select on the GPU; transfer only one [token,status] pair for control flow.

    The caller may separately copy logits/traces for diagnostics. Python loop,
    stop policy and token input preparation are not a production-native runtime.
    """
    plan.run(logits.view(1, -1), out=out, workspace=workspace)
    token, status = out.cpu()[0].tolist()
    if status != 0 or not 0 <= token < plan.vocab_size:
        raise ValueError("Native greedy rejected nonfinite logits or invalid token")
    return token


def snapshot(session):
    return dict(
        gdn=[
            None
            if s is None
            else (s.history.detach().cpu().clone(), s.recurrent.detach().cpu().clone())
            for s in session._states
        ],
        kv=[
            None
            if s is None
            else (
                s.keys[: session.position].detach().cpu().clone(),
                s.values[: session.position].detach().cpu().clone(),
            )
            for s in session._kv
        ],
    )


def compare_state(actual, reference):
    results = []
    for kind in ("gdn", "kv"):
        for layer, (a, b) in enumerate(zip(actual[kind], reference[kind])):
            if a is not None:
                for index, (x, y) in enumerate(zip(a, b)):
                    results.append(
                        dict(kind=kind, layer=layer, tensor=index, **metrics(x, y))
                    )
    return results


def compare_trace(actual, reference):
    if len(actual) != len(reference):
        raise ValueError("Trace length mismatch")
    results = []
    for a, b in zip(actual, reference):
        if (a["position"], a["layer"]) != (b["position"], b["layer"]):
            raise ValueError("Trace ordering mismatch")
        results.append(
            dict(
                position=a["position"],
                layer=a["layer"],
                **metrics(a["hidden"], b["hidden"]),
            )
        )
    return results


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(16 * 2**20):
            value.update(block)
    return value.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--blocks-library", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--tokenizer-file", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--run-cpu-reference", action="store_true")
    parser.add_argument(
        "--sampler", choices=("native-greedy", "cpu-greedy"), default="native-greedy"
    )
    args = parser.parse_args(argv)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = perf_counter()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    case = json.loads(Path(args.case).read_text())
    ids = case["prompt_token_ids"]
    new_tokens = case["max_new_tokens"]
    if type(new_tokens) is not int or not 1 <= new_tokens <= 16:
        raise ValueError("Use 1..16 diagnostic tokens")
    if digest(Path(args.tokenizer_file)) != case["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash mismatch")
    artifact = NInferArtifact(args.artifact)
    weights = QwenQ8Weights.from_ninfer(artifact)
    print("Hashing complete original-derived artifact", flush=True)
    if digest(artifact.path) != args.expected_sha256:
        raise ValueError("Artifact hash mismatch")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    native = StreamingQwenModel.from_q8(
        weights, CudaBackend(args.blocks_library), dtype=dtype, device="cuda:0"
    )
    report = dict(
        scope=__doc__,
        case=case,
        artifact_id=artifact.artifact_id,
        artifact_sha256=args.expected_sha256,
        blocks_library=args.blocks_library,
        dtype=str(dtype),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        native_numerical_layer_operators=True,
        production_native_runtime_qualified=False,
        sampler=(
            "Owned native GPU greedy selector; 16-byte token/status host copy. "
            "Python stop loop and diagnostic full-logit copies remain."
            if args.sampler == "native-greedy"
            else "Explicit host CPU Torch greedy argmax, not owned native sampling"
        ),
        sampler_policy="Greedy with lowest-index ties; nonfinite logits reject, no stochastic sampling",
        native_greedy_selector=args.sampler == "native-greedy",
        offload="Explicit serialized layer/vocabulary streaming with staged KV copies",
        native_steps=[],
        reference_steps=[],
        full_original_bf16_quality_qualified=False,
        performance_qualified=False,
        counters="No achieved occupancy/SM/DRAM/cache/stall counters; no profiler or engine timing. RmProfilingAdminOnly=1 on this host.",
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def observer(event):
        print(json.dumps(event), flush=True)

    save()
    session = native.create_session(
        len(ids) + new_tokens, max_tokens=max(1, len(ids)), observer=observer
    )
    native_logits = []
    generated = []
    selector = (
        prepare_greedy(
            native.backend,
            rows=1,
            vocab_size=native.config["vocab_size"],
            dtype=torch.float32,
            device=native.device,
        )
        if args.sampler == "native-greedy"
        else None
    )
    selection = (
        torch.empty(1, 2, dtype=torch.int64, device=native.device) if selector else None
    )
    selection_scratch = (
        torch.empty(selector.workspace_bytes, dtype=torch.uint8, device=native.device)
        if selector
        else None
    )
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(new_tokens):
            trace = []
            device_logits = session.forward(
                ids if step == 0 else [generated[-1]], trace=trace
            )
            token = (
                planned_greedy(device_logits, selector, selection, selection_scratch)
                if selector
                else cpu_greedy(device_logits)
            )
            # Explicit diagnostic copy, not used to select a native token.
            logits = device_logits.cpu()
            generated.append(token)
            native_logits.append(logits)
            torch.save(
                dict(logits=logits, trace=trace, state=snapshot(session)),
                out / f"native-step-{step}.pt",
            )
            top = logits.topk(10)
            report["native_steps"].append(
                dict(
                    step=step,
                    position=session.position,
                    token=token,
                    top_ids=top.indices.tolist(),
                    top_logits=top.values.tolist(),
                    cache_bytes=session.cache_bytes,
                    workspace_bytes=session.workspace_bytes,
                    finite=True,
                )
            )
            save()
            if token in case["eos_token_ids"]:
                break
        report["native_generated_ids"] = generated
        report["native_full_forward_completed"] = True
        report["native_peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["native_peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        session.close()
    del session, native
    gc.collect()
    torch.cuda.empty_cache()
    save()
    if args.run_cpu_reference:
        # Conservatively account for explicit full-matrix FP64 decode scratch
        # and its FP32 reference storage, not just one resident encoded matrix.
        available = (
            int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        peak = max(
            32 * obj["shape"][0] * ((obj["shape"][1] + 127) // 128 * 128)
            for obj in artifact.objects.values()
            if obj["format"] == "q8_g32_fp16"
        )
        report["cpu_reference_memory_preflight"] = dict(
            available_bytes=available,
            conservative_conversion_scratch_bytes=peak,
            reserve_bytes=8 * 2**30,
        )
        if available < peak + 8 * 2**30:
            save()
            raise MemoryError(
                "CPU reference conversion scratch would exceed available RAM"
            )
        reference = StreamingQwenModel.from_q8(
            weights, TorchBackend(), dtype=dtype, device="cpu"
        )
        s = reference.create_session(
            len(ids) + new_tokens, max_tokens=max(1, len(ids)), observer=observer
        )
        try:
            for step in range(len(generated)):
                trace = []
                logits = s.forward(
                    ids if step == 0 else [generated[step - 1]], trace=trace
                ).cpu()
                saved = torch.load(
                    out / f"native-step-{step}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                item = dict(
                    step=step,
                    position=s.position,
                    logits=logits_comparison(native_logits[step], logits),
                    layers=compare_trace(saved["trace"], trace),
                    state=compare_state(saved["state"], snapshot(s)),
                )
                torch.save(
                    dict(logits=logits, trace=trace), out / f"reference-step-{step}.pt"
                )
                report["reference_steps"].append(item)
                save()
            report["teacher_forced_cpu_reference_completed"] = True
            report["numerical_agreement_gate_passed"] = all(
                s["logits"]["passed"] for s in report["reference_steps"]
            )
        finally:
            s.close()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["qualification_wall_seconds_including_hash_upload_reference_diagnostics"] = (
        perf_counter() - start
    )
    report["process_cpu_seconds_including_all_io_and_reference"] = (
        usage.ru_utime + usage.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime
    )
    report["peak_process_rss_bytes_linux"] = usage.ru_maxrss * 1024
    save()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("reference_steps",)}, indent=2
        )
    )
    if args.run_cpu_reference and not report["numerical_agreement_gate_passed"]:
        raise AssertionError(
            "Fixed full-model numerical diagnostic gate failed; report preserved"
        )


if __name__ == "__main__":
    main()
