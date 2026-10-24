"""Full trained Q8 native-runtime qualification against pinned previous evidence.

This uses an independent C++ forward/control/cache implementation, but shares
checkpoint metadata and small-constant loading helpers. Prior CPU reference is
the same represented Q8/FP16 profile, NOT original-BF16/upstream/corpus quality.
Diagnostic forward/snapshots are explicit; actual generation is one native call.
"""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import resource
from time import perf_counter
from unittest.mock import patch

import torch

from qualify_qwen_streaming import digest, logits_comparison, metrics
from venturi.models import NativeQwenModel, NativeQwenPipelineModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def bytes_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def compare_saved(candidate, previous):
    rows = []
    if len(candidate["trace"]) != len(previous["trace"]):
        raise AssertionError("Native trace length changed")
    for a, b in zip(candidate["trace"], previous["trace"]):
        if (a["position"], a["layer"]) != (b["position"], b["layer"]):
            raise AssertionError("Native trace order changed")
        rows.append(
            dict(
                kind="hidden",
                layer=a["layer"],
                position=a["position"],
                bytes_equal=bytes_equal(a["hidden"], b["hidden"]),
                **metrics(a["hidden"], b["hidden"]),
            )
        )
    for kind in ("gdn", "kv"):
        a, b = candidate["state"][kind], previous["state"][kind]
        if len(a) != len(b):
            raise AssertionError("Native state slots changed")
        for layer, (u, v) in enumerate(zip(a, b)):
            if (u is None) != (v is None):
                raise AssertionError("Native state layer kind changed")
            if u is not None:
                if len(u) != len(v):
                    raise AssertionError("Native state tensor count changed")
                for tensor, (x, y) in enumerate(zip(u, v)):
                    rows.append(
                        dict(
                            kind=kind,
                            layer=layer,
                            tensor=tensor,
                            bytes_equal=bytes_equal(x, y),
                            **metrics(x, y),
                        )
                    )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--blocks-library", required=True)
    p.add_argument("--previous-directory", required=True)
    p.add_argument("--previous-report-sha256", required=True)
    p.add_argument("--tokenizer-file", required=True)
    p.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    p.add_argument("--weight-mode", choices=("streaming", "resident"), required=True)
    p.add_argument("--tensor-parallel-device")
    p.add_argument(
        "--tensor-parallel-transport", choices=("peer", "host"), default="peer"
    )
    p.add_argument("--pipeline-device")
    p.add_argument("--pipeline-split-layer", type=int)
    p.add_argument("--output-directory", required=True)
    args = p.parse_args()
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = perf_counter()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    prevdir = Path(args.previous_directory)
    prevpath = prevdir / "report.json"
    if digest(prevpath) != args.previous_report_sha256:
        raise ValueError("Pinned previous report changed")
    prev = json.loads(prevpath.read_text())
    if (
        not prev["teacher_forced_cpu_reference_completed"]
        or not prev["numerical_agreement_gate_passed"]
    ):
        raise ValueError("Previous CPU reference incomplete or rejected")
    if prev["artifact_sha256"] != args.expected_sha256:
        raise ValueError("Reference must use the same artifact")
    case = prev["case"]
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if prev["dtype"] != str(dtype):
        raise ValueError("Reference activation profile differs")
    if digest(Path(args.tokenizer_file)) != case["tokenizer_sha256"]:
        raise ValueError("Original tokenizer hash changed")
    print("Hashing complete Q8 artifact before native mapping", flush=True)
    if digest(Path(args.artifact)) != args.expected_sha256:
        raise ValueError("Artifact hash changed")
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    report = dict(
        scope=__doc__,
        artifact_sha256=args.expected_sha256,
        previous_report_sha256=args.previous_report_sha256,
        case=case,
        dtype=str(dtype),
        weight_mode=args.weight_mode,
        tensor_parallel_device=args.tensor_parallel_device,
        tensor_parallel_transport=args.tensor_parallel_transport,
        pipeline_device=args.pipeline_device,
        pipeline_split_layer=args.pipeline_split_layer,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        numerical_and_control_runtime="Owned C++/CUDA; no Python layer loop or generation loop",
        loading="Python metadata and small-constant preparation; native copies own constants and mmap Q8 planes",
        whole_model_graphs_qualified=False,
        continuous_batching_paged_kv_qualified=False,
        full_original_bf16_corpus_quality_qualified=False,
        independent_upstream_architecture_qualified=False,
        matched_engine_performance_qualified=False,
        steps=[],
        previous_snapshot_sha256={},
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(
        "Creating explicit native model; bounded encoded validation, not dense expansion",
        flush=True,
    )
    common = dict(
        weight_mode=args.weight_mode,
        dtype=dtype,
        device="cuda:0",
        max_tokens=len(case["prompt_token_ids"]),
    )
    if args.pipeline_device:
        if args.tensor_parallel_device:
            raise ValueError("Select explicit TP or PP, not both")
        model = NativeQwenPipelineModel.from_q8(
            weights,
            CudaBackend(args.blocks_library),
            **common,
            pipeline_device=args.pipeline_device,
            split_layer=args.pipeline_split_layer,
            transport="host",
        )
    else:
        model = NativeQwenModel.from_q8(
            weights,
            CudaBackend(args.blocks_library),
            **common,
            tensor_parallel_device=args.tensor_parallel_device,
            tensor_parallel_transport=args.tensor_parallel_transport,
        )
    s = model.create_session(len(case["prompt_token_ids"]) + case["max_new_tokens"])
    report["initial_native_memory"] = s.info()
    save()
    try:
        for step, token in enumerate(prev["native_generated_ids"]):
            trace = []
            logits = s.forward(
                case["prompt_token_ids"]
                if step == 0
                else [prev["native_generated_ids"][step - 1]],
                trace=trace,
            )
            current = dict(logits=logits, trace=trace, state=s.snapshot())
            torch.save(current, out / f"native-step-{step}.pt")
            nativepath = prevdir / f"native-step-{step}.pt"
            cpupath = prevdir / f"reference-step-{step}.pt"
            report["previous_snapshot_sha256"][nativepath.name] = digest(nativepath)
            report["previous_snapshot_sha256"][cpupath.name] = digest(cpupath)
            old = torch.load(nativepath, map_location="cpu", weights_only=True)
            cpu = torch.load(cpupath, map_location="cpu", weights_only=True)
            # Bind prior CPU logits to its already-pinned report, not a filename alone.
            if (
                logits_comparison(old["logits"], cpu["logits"])
                != prev["reference_steps"][step]["logits"]
            ):
                raise ValueError("Previous saved logits no longer match their report")
            layers = compare_saved(current, old)
            item = dict(
                step=step,
                position=s.position,
                expected_token=token,
                logits_vs_cpu=logits_comparison(logits, cpu["logits"]),
                logits_bytes_equal_previous_native=bytes_equal(logits, old["logits"]),
                layers_and_state=layers,
                all_hidden_state_bytes_equal=all(r["bytes_equal"] for r in layers),
                native_memory=s.info(),
            )
            report["steps"].append(item)
            save()
            print(
                "Native diagnostic step",
                step,
                "complete, fixed CPU gate:",
                item["logits_vs_cpu"]["passed"],
                "exact native parity:",
                item["all_hidden_state_bytes_equal"],
                flush=True,
            )
            del current, old, cpu, trace, logits, layers
            gc.collect()
    finally:
        s.close()
    # A separate request demonstrates the real C++ generation loop, not the
    # above Python diagnostic sequence. No Torch numerical operation is allowed.
    s = model.create_session(len(case["prompt_token_ids"]) + case["max_new_tokens"])

    def forbidden(*a, **k):
        raise AssertionError(
            "Native generation called numerical Torch/Python model code"
        )

    try:
        with ExitStack() as stack:
            for name in (
                "mm",
                "matmul",
                "bmm",
                "argmax",
                "cos",
                "sin",
                "empty",
                "zeros",
            ):
                stack.enter_context(patch.object(torch, name, forbidden))
            for name in ("to", "cpu", "float", "argmax", "matmul", "__matmul__"):
                stack.enter_context(patch.object(torch.Tensor, name, forbidden))
            stack.enter_context(patch.object(QwenQ8Weights, "read_linear", forbidden))
            generated = s.generate(
                case["prompt_token_ids"],
                max_new_tokens=case["max_new_tokens"],
                eos_token_ids=case["eos_token_ids"],
            )
        report["native_generated_ids"] = generated
        report["native_generation_position"] = s.position
        report["native_generation_memory"] = s.info()
        report["native_cpp_generation_without_torch_math_completed"] = True
        report["generation_matches_previous"] = (
            generated == prev["native_generated_ids"]
        )
    finally:
        s.close()
        model.close()
    report["all_fixed_cpu_logit_gates_passed"] = all(
        s["logits_vs_cpu"]["passed"] for s in report["steps"]
    )
    report["all_native_logit_hidden_state_bytes_match_previous"] = all(
        s["logits_bytes_equal_previous_native"] and s["all_hidden_state_bytes_equal"]
        for s in report["steps"]
    )
    report["native_full_runtime_case_passed"] = (
        report["generation_matches_previous"]
        and report["all_fixed_cpu_logit_gates_passed"]
        and report["all_native_logit_hidden_state_bytes_match_previous"]
    )
    final = resource.getrusage(resource.RUSAGE_SELF)
    report["qualification_wall_seconds_including_hash_loading_diagnostics"] = (
        perf_counter() - start
    )
    report["qualification_process_cpu_seconds"] = (
        final.ru_utime + final.ru_stime - usage.ru_utime - usage.ru_stime
    )
    report["peak_process_rss_bytes_linux_including_mapped_page_residency"] = (
        final.ru_maxrss * 1024
    )
    save()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("steps", "case")}, indent=2
        ),
        flush=True,
    )
    if not report["native_full_runtime_case_passed"]:
        raise AssertionError("Native full runtime case failed; evidence preserved")


if __name__ == "__main__":
    main()
