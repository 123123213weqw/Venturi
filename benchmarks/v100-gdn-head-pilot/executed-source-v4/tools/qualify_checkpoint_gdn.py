"""Remote-only Qwen-artifact vocabulary and actual-geometry GDN qualification.

No external model forward or model downloads. GDN uses synthetic represented
inputs and the artifact's dimensions; Q8 vocabulary tests use actual checkpoint
bytes. Report primitive calls/s, never model TPS. Timing is unprofiled ABBA with
matching precision/input/output and explicit FP32 tiled Torch references.
"""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import resource
import statistics
import subprocess
from time import perf_counter, time

import torch

from benchmark_qwen_gpu import FIELDS, telemetry
from run_qwen_validation import idle_gpu
from venturi.ninfer import NInferArtifact
from venturi.ops import (
    CudaBackend,
    NInferVocabulary,
    Q8VocabularyWeights,
    ResidentQ8Vocabulary,
    TorchBackend,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def phase(call, seconds):
    start, wall_start, cpu_start = time(), perf_counter(), cpu_seconds()
    count, event_ms = 0, 0.0
    while perf_counter() - wall_start < seconds:
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        for _ in range(10):
            call()
        b.record()
        b.synchronize()
        event_ms += a.elapsed_time(b)
        count += 10
    wall, cpu = perf_counter() - wall_start, cpu_seconds() - cpu_start
    return dict(
        mode="single",
        start_unix=start,
        end_unix=time(),
        calls=count,
        wall_seconds=wall,
        process_cpu_seconds=cpu,
        process_cpu_percent=100 * cpu / wall,
        wall_us_per_call=wall / count * 1e6,
        event_us_per_call=event_ms / count * 1000,
        calls_per_second=count / wall,
    )


def timed_pair(calls, output, seconds):
    for call in calls.values():
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    phases = []
    for name in ("torch-reference", "native", "native", "torch-reference"):
        measured = phase(calls[name], seconds)
        measured["condition"] = name
        phases.append(measured)
        write(output, phases)
    return phases


def comparison(phases):
    return {
        name: {
            key: statistics.median(p[key] for p in phases if p["condition"] == name)
            for key in (
                "wall_us_per_call",
                "event_us_per_call",
                "process_cpu_percent",
                "calls_per_second",
            )
        }
        for name in ("torch-reference", "native")
    }


def bf16_check(actual, ideal, *, floor=2e-5):
    actual, ideal = actual.cpu().double(), ideal.cpu().double()
    error = (actual - ideal.bfloat16().double()).abs()
    bound = 2 * torch.finfo(torch.bfloat16).eps * ideal.abs() + floor
    failed = error > bound
    return dict(
        passed=not bool(failed.any()),
        max_rounded_abs_error=error.max().item(),
        max_unrounded_abs_error=(actual - ideal).abs().max().item(),
        elements=actual.numel(),
        failed_elements=int(failed.sum()),
        criterion=f"abs(actual-rounded_FP64) <= 2*BF16_eps*abs(FP64) + {floor}",
        first_failure_indices=failed.nonzero()[:8].tolist(),
    )


def qualify_vocabulary_bindings(artifact, backend):
    """Exercise the public two-binding API with actual token-embedding input."""
    bound = NInferVocabulary.upload_from_ninfer(artifact, backend)
    ids = [1, 42, bound.embedding.vocab_size - 1]
    lookup = bound.embedding.prepare_embedding(ids)
    hidden = torch.empty(
        len(ids), bound.embedding.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    lookup.run(out=hidden)
    source = bound.object_ids[0]
    expected = torch.tensor([artifact.decode_row(source, i) for i in ids]).bfloat16()
    assert torch.equal(hidden.cpu(), expected)
    logits = torch.empty(1, bound.head.vocab_size, dtype=torch.bfloat16, device="cuda")
    bound.head.prepare(tokens=1).run(hidden[1:2], out=logits)
    assert bool(torch.isfinite(logits).all())
    rows = sorted(
        set(
            [
                0,
                1,
                42,
                bound.head.vocab_size - 1,
                *logits[0].topk(10).indices.cpu().tolist(),
            ]
        )
    )
    head_rows = torch.tensor(
        [artifact.decode_row(bound.object_ids[1], i) for i in rows], dtype=torch.float64
    )
    ideal = hidden[1:2].cpu().double() @ head_rows.T
    checked = bf16_check(logits[:, rows], ideal, floor=2e-4)
    assert checked["passed"], checked
    return dict(
        object_ids=bound.object_ids,
        embedding_shape=[bound.embedding.vocab_size, bound.embedding.hidden_size],
        head_shape=[bound.head.vocab_size, bound.head.hidden_size],
        resident_weight_bytes=bound.storage_nbytes,
        tied_storage=bound.embedding is bound.head,
        embedding_exact=True,
        head_logits_finite=True,
        head_sample_rows=rows,
        head_sample_qualification=checked,
        scope="Public vocabulary binding composition only; embedding input is not a decoder hidden state",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--blocks-library", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128])
    args = parser.parse_args(argv)
    if args.seconds < 2 or not args.tokens or any(t <= 0 for t in args.tokens):
        raise ValueError("Use >=2 second phases and positive token counts")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    write(out / "background.json", {"compute_processes": idle_gpu()})
    native = CudaBackend(args.blocks_library)
    artifact = NInferArtifact(args.checkpoint)
    config = artifact.manifest["components"]["text"]["config"]
    if config.get("model_type") != "qwen3_5_text":
        raise ValueError("This qualification expects the current qwen3_5_text artifact")
    write(
        out / "artifact-config.json",
        dict(
            artifact_id=artifact.artifact_id,
            path=str(artifact.path),
            config=config,
            vocabulary_bindings={
                name: artifact.objects[artifact.manifest["bindings"][name]["object"]]
                for name in ("text/token_embedding", "text/output_head")
            },
        ),
    )
    report = dict(
        scope="Independent primitives only, NOT full Qwen generation/quality or model TPS",
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        native_library=str(native.library),
        cases=[],
        missing_integration=[
            "GDN causal convolution and per-request convolution history",
            "GDN gate projections/transforms and gated output norm",
            "hybrid layer scheduler/adapter",
            "Qwen zero-centered (1 + weight) normalization convention",
            "packed multipart Q4 projection binding views",
            "Qwen Q/K norm and attention output gate",
            "partial/multimodal RoPE checkpoint conventions",
        ],
        profiler="No profiler in throughput runs. Hardware counters unavailable under RmProfilingAdminOnly=1; achieved occupancy/bandwidth not measured.",
    )
    spec = spec_from_file_location(
        "gdn_qualification_oracle",
        Path(__file__).resolve().parents[1] / "tests/test_gdn.py",
    )
    oracle = module_from_spec(spec)
    spec.loader.exec_module(oracle)
    waves = []
    with (out / "telemetry.csv").open("w") as sink:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(FIELDS),
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ],
            stdout=sink,
            stderr=subprocess.DEVNULL,
        )
        try:
            for tokens in args.tokens:
                print(f"GDN T={tokens}, checkpoint geometry", flush=True)
                hq, hv, dk, dv = (
                    config[k]
                    for k in (
                        "linear_num_key_heads",
                        "linear_num_value_heads",
                        "linear_key_head_dim",
                        "linear_value_head_dim",
                    )
                )
                q, k, v, g, beta, state = oracle.gdn_inputs(
                    tokens, hq, hv, dk, dv, torch.bfloat16, "cuda"
                )
                ideal, ideal_state = oracle.gdn_oracle(q, k, v, g, beta, state)
                plans = {
                    name: backend.prepare_gdn(
                        tokens=tokens,
                        qk_heads=hq,
                        value_heads=hv,
                        key_dim=dk,
                        value_dim=dv,
                    )
                    for name, backend in (
                        ("native", native),
                        ("torch-reference", TorchBackend()),
                    )
                }
                outputs = {
                    name: (torch.empty_like(v), torch.empty_like(state))
                    for name in plans
                }
                graphs = {}
                case = dict(
                    name=f"gdn-t{tokens}",
                    tokens=tokens,
                    qk_heads=hq,
                    value_heads=hv,
                    key_dim=dk,
                    value_dim=dv,
                    input_dtype="BF16",
                    gate_state_dtype="FP32",
                    state_bytes=state.numel() * 4,
                    input_source="Synthetic represented inputs; artifact geometry, not checkpoint hidden states",
                    timing="Fixed input and non-mutating initial state; captured graph replay including Python replay wrapper, not autoregressive rollout",
                )
                checks = {}
                for name, plan in plans.items():
                    target, final = outputs[name]
                    plan.run(q, k, v, g, beta, state=state, out=target, state_out=final)
                    check = bf16_check(target, ideal)
                    state_error = (final.cpu().double() - ideal_state).abs()
                    state_bound = 2e-6 + 2e-4 * ideal_state.abs()
                    check["fp32_state_passed"] = bool(
                        (state_error <= state_bound).all()
                    )
                    check["fp32_state_max_abs_error"] = state_error.max().item()
                    checks[name] = check
                    case["qualification"] = checks
                    write(out / (case["name"] + "-qualification.json"), case)
                    assert check["passed"] and check["fp32_state_passed"], checks
                    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            plan.run(
                                q,
                                k,
                                v,
                                g,
                                beta,
                                state=state,
                                out=target,
                                state_out=final,
                            )
                        with torch.cuda.graph(graph, stream=stream):
                            plan.run(
                                q,
                                k,
                                v,
                                g,
                                beta,
                                state=state,
                                out=target,
                                state_out=final,
                            )
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    graphs[name] = graph
                case["phases"] = timed_pair(
                    {name: graph.replay for name, graph in graphs.items()},
                    out / (case["name"] + "-phases.json"),
                    args.seconds,
                )
                case["medians"] = comparison(case["phases"])
                waves.extend(case["phases"])
                report["cases"].append(case)
                write(out / "results.json", report)
                del graphs, outputs, plans, q, k, v, g, beta, state, ideal, ideal_state
                torch.cuda.synchronize()
            # Real checkpoint vocabulary bytes: bind and test each matrix
            # independently, releasing the previous matrix before the next one.
            for binding in ("text/token_embedding", "text/output_head"):
                print("binding", binding, flush=True)
                weights = Q8VocabularyWeights.from_ninfer(artifact, binding)
                resident = weights.upload(native)
                reference = ResidentQ8Vocabulary(
                    TorchBackend(),
                    resident._codes,
                    resident._scales,
                    resident.hidden_size,
                )
                ids = [
                    1,
                    42,
                    resident.vocab_size - 1,
                    min(248070, resident.vocab_size - 1),
                ]
                lookup = resident.prepare_embedding(ids)
                embedding = torch.empty(
                    len(ids), resident.hidden_size, dtype=torch.bfloat16, device="cuda"
                )
                lookup.run(out=embedding)
                # Existing independent artifact decoder validates the storage
                # binding, including signs, group scales and payload offsets.
                obj = artifact.manifest["bindings"][binding]["object"]
                expected_rows = torch.tensor([artifact.decode_row(obj, i) for i in ids])
                assert torch.equal(embedding.cpu(), expected_rows.bfloat16())
                case = dict(
                    name=binding.replace("/", "-"),
                    binding=binding,
                    format="q8_g32_fp16",
                    shape=[resident.vocab_size, resident.hidden_size],
                    resident_bytes=resident.storage_nbytes,
                    embedding_ids=ids,
                    embedding_bit_exact_to_independent_decoder=True,
                    timing="Eager prepared head only, fixed single BF16 hidden vector; no ID upload, checkpoint IO, model decode or full-model TPS",
                )
                if binding.endswith("output_head"):
                    x = embedding[1:2]
                    targets = {
                        name: torch.empty(
                            1, resident.vocab_size, dtype=torch.bfloat16, device="cuda"
                        )
                        for name in ("native", "torch-reference")
                    }
                    plans = {
                        "native": resident.prepare(tokens=1),
                        "torch-reference": reference.prepare(tokens=1),
                    }
                    for name, plan in plans.items():
                        plan.run(x, out=targets[name])
                    # All vocabulary logits, bounded-row independent CPU FP64.
                    x64 = x.cpu().double()
                    ideal = torch.empty(1, resident.vocab_size, dtype=torch.float64)
                    for start in range(0, resident.vocab_size, 512):
                        end = min(start + 512, resident.vocab_size)
                        dense = weights.codes[start:end].double() * weights.scales[
                            start:end
                        ].double().repeat_interleave(32, -1)
                        ideal[:, start:end] = x64 @ dense[:, : resident.hidden_size].T
                    case["qualification"] = {
                        name: bf16_check(target, ideal, floor=2e-4)
                        for name, target in targets.items()
                    }
                    write(out / (case["name"] + "-qualification.json"), case)
                    assert all(c["passed"] for c in case["qualification"].values()), (
                        case
                    )
                    calls = {
                        name: (
                            lambda plan=plan, target=targets[name]: plan.run(
                                x, out=target
                            )
                        )
                        for name, plan in plans.items()
                    }
                    case["phases"] = timed_pair(
                        calls, out / (case["name"] + "-phases.json"), args.seconds
                    )
                    case["medians"] = comparison(case["phases"])
                    waves.extend(case["phases"])
                report["cases"].append(case)
                write(out / "results.json", report)
                torch.cuda.synchronize()
                del weights, resident, reference, lookup, embedding
            report["public_bindings_api"] = qualify_vocabulary_bindings(
                artifact, native
            )
            write(out / "public-bindings-api.json", report["public_bindings_api"])
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
    report["telemetry"] = {
        condition: telemetry(
            out / "telemetry.csv", [p for p in waves if p["condition"] == condition]
        )["single"]
        for condition in ("native", "torch-reference")
    }
    for case in report["cases"]:
        if "phases" in case:
            case["telemetry"] = {
                condition: telemetry(
                    out / "telemetry.csv",
                    [p for p in case["phases"] if p["condition"] == condition],
                )["single"]
                for condition in ("native", "torch-reference")
            }
    write(out / "results.json", report)
    print(
        json.dumps(
            {c["name"]: c.get("medians", "lookup passed") for c in report["cases"]},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
