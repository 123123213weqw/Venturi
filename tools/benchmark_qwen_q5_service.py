"""Same-instance SM70/Q5 correctness and loopback service A/B, never compilation.

A=bounded Q5 decode+TC cuBLAS; B=startup-calibrated native head-only routing. Native
cohort generate owns the entire token loop. Fixed pretokenized prompts, greedy,
no EOS early exit/prefix reuse/MTP/graphs. HTTP cohort wall is global, not a
sum of concurrent request durations. Driver runs in a separate CPU process.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen


def validate_cohort(responses, count):
    """Reject admission/backlog differences rather than attributing them to kernels."""
    if len(responses) != count:
        raise AssertionError("Missing HTTP responses")
    cohorts = {r["response"]["batch_cohort_id"] for r in responses}
    if len(cohorts) != 1 or any(
        r["response"]["batch_request_count"] != count for r in responses
    ):
        raise AssertionError(
            "HTTP wave split across native cohorts; reject unmatched TPS"
        )


def client(port, count, prompt_length, output_tokens, prompts):
    barrier = threading.Barrier(count + 1)

    def one(slot):
        prompt = (prompts[slot] * ((prompt_length + 15) // 16))[:prompt_length]
        payload = dict(
            prompt=prompt,
            n_predict=output_tokens,
            id_slot=slot,
            stream=False,
            cache_prompt=False,
            ignore_eos=True,
            temperature=0,
            repeat_penalty=1,
        )
        request = Request(
            f"http://127.0.0.1:{port}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        barrier.wait()
        began = time.perf_counter()
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
        assert len(result["tokens"]) == output_tokens
        return dict(latency_seconds=time.perf_counter() - began, response=result)

    cpu = time.process_time()
    with ThreadPoolExecutor(count) as pool:
        futures = [pool.submit(one, slot) for slot in range(count)]
        began = time.perf_counter()
        barrier.wait()
        responses = [f.result() for f in futures]
        wall = time.perf_counter() - began
    validate_cohort(responses, count)
    return dict(
        wall_seconds=wall,
        client_cpu_seconds=time.process_time() - cpu,
        output_tokens=count * output_tokens,
        output_tps=count * output_tokens / wall,
        requests=responses,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact")
    p.add_argument("--library")
    p.add_argument("--qwen-library")
    p.add_argument("--prompts", required=True)
    p.add_argument("--output")
    p.add_argument("--client", action="store_true")
    p.add_argument("--port", type=int, default=18015)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--prompt-length", type=int, default=16)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--correctness-report",
        help="Reuse completed native gates only after exact checkpoint/DSO/route/condition identity checks",
    )
    p.add_argument("--profile-only", action="store_true")
    p.add_argument(
        "--verify-long-batch",
        action="store_true",
        help="Additional 128-token / 8-slot forced state check before profiling",
    )
    args = p.parse_args()
    prompts = json.loads(Path(args.prompts).read_text())["prompt_token_ids"]
    if args.client:
        print(
            json.dumps(
                client(
                    args.port,
                    args.count,
                    args.prompt_length,
                    args.output_tokens,
                    prompts,
                )
            )
        )
        return

    import torch
    from venturi.models import QwenWeights, NativeQwenModel
    from venturi.models.qwen.native_runtime import QwenRuntime
    from venturi.ops import CudaBackend
    from benchmark_q5_adaptive import Telemetry
    from serve_native_benchmark import CompletionBatcher, request_values

    torch.set_num_threads(4)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope=__doc__,
        completed=False,
        instrumented=args.profile_only,
        phases=[],
        correctness=[],
        waves=[],
        thresholds=dict(
            logits_max_abs=0.25, logits_relative_l2=0.01, state_relative_l2=0.0
        ),
        conditions=dict(
            model="Qwen3.8-27B text, 64 layers",
            precision="Q5 G64 / FP16 X,W,Y / FP32 accumulation",
            device="cuda:0 V100",
            context_capacity=192,
            projection_rows=32,
            slots=8,
            output_tokens=args.output_tokens,
            warmup="one discarded cohort per route/shape, plus whole-model correctness warmup",
            timing="unprofiled HTTP global cohort wall; includes prompt, decode, admission, JSON/network; excludes load/calibration",
            batch_window_ms=None,
            admission="Fixed complete cohort, immediate when ready; 5-second collection deadline",
            minimum_gain=0.03,
            clocks="unlocked",
            prefix_reuse=False,
            mtp=False,
            cuda_graphs=False,
            input="16-token pinned text IDs; 128-token cases repeat the same 16 IDs eight times, not a natural long document",
        ),
    )

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def phase(name):
        report["phases"].append(dict(name=name, unix=time.time()))
        print(name, flush=True)
        save()

    def metric(a, b):
        a, b = a.float(), b.float()
        delta = a - b
        return dict(
            elementwise_exact=bool(torch.equal(a, b)),
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            max_abs=float(delta.abs().max()),
            relative_l2=float(delta.norm() / b.norm().clamp_min(1e-12)),
            rms=float(delta.square().mean().sqrt()),
        )

    def states(actual, reference):
        result = {}
        for kind in ("gdn", "kv"):
            rows = []
            for layer, (a, b) in enumerate(zip(actual[kind], reference[kind])):
                if a is not None:
                    for tensor, (x, y) in enumerate(zip(a, b)):
                        rows.append(dict(layer=layer, tensor=tensor, **metric(x, y)))
            result[kind] = rows
        return result

    model = batch = admission = server = None
    with Telemetry() as telemetry:
        try:
            phase("load")
            weights = QwenWeights.open(args.artifact)
            report["artifact"] = dict(
                path=str(Path(args.artifact).resolve()),
                bytes=Path(args.artifact).stat().st_size,
                provenance=weights.provenance,
                config=weights.config,
            )
            # Sequential digest outside all timed work. Avoid a 17 GiB read_bytes allocation.
            digest = hashlib.sha256()
            with open(args.artifact, "rb") as source:
                for data in iter(lambda: source.read(8 * 1024**2), b""):
                    digest.update(data)
            report["artifact"]["sha256"] = digest.hexdigest()
            report["libraries"] = {
                key: dict(
                    path=value,
                    sha256=hashlib.sha256(Path(value).read_bytes()).hexdigest(),
                )
                for key, value in (("ops", args.library), ("qwen", args.qwen_library))
            }
            report["prompts"] = json.loads(Path(args.prompts).read_text())
            model = NativeQwenModel.from_q8(
                weights,
                QwenRuntime(args.qwen_library, CudaBackend(args.library)),
                weight_mode="resident",
                dtype=torch.float16,
                device="cuda:0",
                max_tokens=32,
                metadata_mode="sealed",
            )
            model.configure_q5_linear("cublas")
            phase("startup-calibration")
            began = time.perf_counter()
            report["calibration"] = model.calibrate_q5_linear(
                [1, 2, 3, 4, 5, 6, 7, 8, 16, 32], repeats=5, iterations=5
            )
            report["calibration_seconds"] = time.perf_counter() - began
            save()
            if args.correctness_report:
                reference_path = Path(args.correctness_report)
                old = json.loads(reference_path.read_text())
                for key in ("artifact", "libraries"):
                    if key == "artifact":
                        assert old[key]["sha256"] == report[key]["sha256"]
                    else:
                        assert {k: v["sha256"] for k, v in old[key].items()} == {
                            k: v["sha256"] for k, v in report[key].items()
                        }
                old_conditions = {
                    k: v
                    for k, v in old["conditions"].items()
                    if k not in ("batch_window_ms", "admission")
                }
                new_conditions = {
                    k: v
                    for k, v in report["conditions"].items()
                    if k not in ("batch_window_ms", "admission")
                }
                assert old_conditions == new_conditions, (
                    "Native correctness conditions changed"
                )
                assert old["prompts"]["prompt_token_ids"] == prompts, (
                    "Native correctness prompt token IDs changed"
                )

                def route_key(row):
                    return row["n"], row["k"], row["tokens"], row["selected"]

                assert old["calibration"]["policy"] == report["calibration"]["policy"]
                assert sorted(
                    map(route_key, old["calibration"]["calibration"])
                ) == sorted(map(route_key, report["calibration"]["calibration"])), (
                    "Calibrated routes changed"
                )
                assert (
                    len(old["correctness"]) == 4 and len(old["batch_correctness"]) == 4
                )
                assert all(
                    row["numerical_gate"] and row["greedy_exact"]
                    for row in old["correctness"]
                )
                assert all(row["numerical_gate"] for row in old["batch_correctness"])
                for key in ("correctness", "batch_correctness", "memory"):
                    report[key] = old[key]
                report["correctness_source"] = dict(
                    path=str(reference_path),
                    sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                    reason="Identical artifact, DSOs, conditions and calibrated routes; only HTTP backlog/admission verification changed",
                )
                phase("reused-identity-checked-native-correctness")
            if not args.profile_only and not args.correctness_report:
                phase("teacher-forced-and-state-correctness")
                session = model.create_session(192)
                try:
                    for case, length in enumerate((16, 32, 96, 128)):
                        prompt = (prompts[case] * ((length + 15) // 16))[:length]
                        model.configure_q5_linear("cublas")
                        session.reset()
                        reference = [session.forward(prompt)]
                        continuation = []
                        for _ in range(31):
                            continuation.append(int(reference[-1].argmax()))
                            reference.append(session.forward([continuation[-1]]))
                        reference_state = session.snapshot()
                        model.configure_q5_linear("adaptive")
                        session.reset()
                        actual = [session.forward(prompt)]
                        for token in continuation:
                            actual.append(session.forward([token]))
                        state = states(session.snapshot(), reference_state)
                        metrics = [
                            dict(
                                position=length + i,
                                top1_equal=int(a.argmax()) == int(b.argmax()),
                                baseline_top2_margin=float(
                                    b.topk(2).values.diff().abs()[0]
                                ),
                                **metric(a, b),
                            )
                            for i, (a, b) in enumerate(zip(actual, reference))
                        ]
                        okay = all(
                            m["finite"]
                            and m["max_abs"] <= 0.25
                            and m["relative_l2"] <= 0.01
                            for m in metrics
                        ) and all(
                            m["finite"] and m["elementwise_exact"]
                            for rows in state.values()
                            for m in rows
                        )
                        row = dict(
                            case=case,
                            prompt_length=length,
                            prompt=prompt,
                            continuation=continuation,
                            logits=metrics,
                            state=state,
                            numerical_gate=okay,
                        )
                        # Independent native greedy loops on exactly the same prompt.
                        generated = {}
                        for route in ("cublas", "adaptive"):
                            model.configure_q5_linear(route)
                            session.reset()
                            generated[route] = session.generate(
                                prompt, max_new_tokens=32, eos_token_ids=[]
                            )
                        row.update(
                            generated=generated,
                            greedy_exact=generated["cublas"] == generated["adaptive"],
                        )
                        report["correctness"].append(row)
                        save()
                    report["memory"] = session.info()
                finally:
                    session.close()
                if not all(r["numerical_gate"] for r in report["correctness"]):
                    raise AssertionError(
                        "Whole-model numerical gate failed; retain evidence, do not publish TPS as qualified"
                    )

            batch = model.create_batch([192] * 8)
            if (
                not args.profile_only and not args.correctness_report
            ) or args.verify_long_batch:
                phase("grouped-correctness")
                report["batch_correctness"] = []
                validation_cases = (
                    [(16, c) for c in (1, 2, 4, 8)]
                    if not args.profile_only
                    else [(128, 8)]
                )
                for length, count in validation_cases:
                    ids = [
                        (prompt * ((length + 15) // 16))[:length]
                        for prompt in prompts[:count]
                    ]
                    model.configure_q5_linear("cublas")
                    batch.reset()
                    reference = [batch.forward(ids)]
                    forced = []
                    for _ in range(31):
                        tokens = reference[-1].argmax(-1).tolist()
                        forced.append(tokens)
                        reference.append(batch.forward([[t] for t in tokens]))
                    selected_slots = sorted({0, count - 1})
                    reference_state = {
                        slot: batch.snapshot(slot) for slot in selected_slots
                    }
                    model.configure_q5_linear("adaptive")
                    batch.reset()
                    actual = [batch.forward(ids)]
                    for tokens in forced:
                        actual.append(batch.forward([[t] for t in tokens]))
                    metrics = [
                        dict(
                            step=i,
                            top1_equal=bool(torch.equal(a.argmax(-1), b.argmax(-1))),
                            **metric(a, b),
                        )
                        for i, (a, b) in enumerate(zip(actual, reference))
                    ]
                    state = {
                        slot: states(batch.snapshot(slot), reference_state[slot])
                        for slot in selected_slots
                    }
                    okay = all(
                        r["finite"]
                        and r["relative_l2"] <= 0.01
                        and r["max_abs"] <= 0.25
                        for r in metrics
                    ) and all(
                        m["finite"] and m["elementwise_exact"]
                        for slots in state.values()
                        for rows in slots.values()
                        for m in rows
                    )
                    row = dict(
                        concurrency=count,
                        prompt_length=length,
                        logits=metrics,
                        state=state,
                        numerical_gate=okay,
                        routing=model.q5_linear_info(),
                    )
                    report["batch_correctness"].append(row)
                    save()
                    if not okay:
                        raise AssertionError("Grouped 32-step logits/state gate failed")

            admission = CompletionBatcher(batch, 8, 0.005)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *unused):
                    pass

                def do_POST(self):
                    try:
                        if self.path != "/completion":
                            raise ValueError("completion route only")
                        body = json.loads(
                            self.rfile.read(int(self.headers["Content-Length"]))
                        )
                        ids, count, slot = request_values(
                            body,
                            8,
                            192,
                            weights.config["vocab_size"],
                            max_prompt_tokens=128,
                        )
                        result = admission.submit(slot, ids, count).result(timeout=600)
                        payload = json.dumps(result).encode()
                        code = 200
                    except Exception as error:
                        payload = json.dumps(dict(error=str(error))).encode()
                        code = 500
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            class Server(ThreadingHTTPServer):
                request_queue_size = 128

            server = Server(("127.0.0.1", args.port), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            phase("profile" if args.profile_only else "unprofiled-http-abba")
            if args.profile_only:
                torch.cuda.cudart().cudaProfilerStart()
            cases = [(16, c) for c in (1, 2, 4, 8)] + [(128, c) for c in (1, 8)]
            if args.profile_only:
                cases = [(16, 1), (16, 8)]
            for length, count in cases:
                # Balanced paired order, two same-instance routes and buffers.
                order = [
                    r
                    for pair in range(args.repeats)
                    for r in (
                        ("cublas", "adaptive")
                        if pair % 2 == 0
                        else ("adaptive", "cublas")
                    )
                ]
                pinned = None
                for warmup, route in [(True, "cublas"), (True, "adaptive")] + [
                    (False, r) for r in order
                ]:
                    # Benchmark-only fixed cohort: real service keeps its
                    # ordinary bounded time-window admission unchanged.
                    admission.slots = count
                    admission.window = 5.0
                    model.configure_q5_linear(route)
                    if args.profile_only:
                        torch.cuda.nvtx.range_push(
                            f"q5/{route}/m{count}/p{length}/warmup{warmup}"
                        )
                    cpu = time.process_time()
                    began = time.time()
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--client",
                        "--prompts",
                        args.prompts,
                        "--port",
                        str(args.port),
                        "--count",
                        str(count),
                        "--prompt-length",
                        str(length),
                        "--output-tokens",
                        str(args.output_tokens),
                    ]
                    run = subprocess.run(
                        cmd, check=True, capture_output=True, text=True, timeout=900
                    )
                    if args.profile_only:
                        torch.cuda.nvtx.range_pop()
                    row = json.loads(run.stdout)
                    row.update(
                        route=route,
                        concurrency=count,
                        prompt_length=length,
                        warmup=warmup,
                        server_cpu_core_seconds=time.process_time() - cpu,
                        started_unix=began,
                        finished_unix=time.time(),
                        routing=model.q5_linear_info(),
                    )
                    tokens = [r["response"]["tokens"] for r in row["requests"]]
                    if pinned is None:
                        pinned = tokens
                    row["tokens_equal_first_cublas_wave"] = tokens == pinned
                    report["waves"].append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "route",
                                    "concurrency",
                                    "prompt_length",
                                    "warmup",
                                    "output_tps",
                                    "tokens_equal_first_cublas_wave",
                                )
                            }
                        ),
                        flush=True,
                    )
            if args.profile_only:
                torch.cuda.cudart().cudaProfilerStop()
            report["summary"] = []
            for length, count in cases:
                groups = {
                    r: [
                        w
                        for w in report["waves"]
                        if not w["warmup"]
                        and w["route"] == r
                        and w["prompt_length"] == length
                        and w["concurrency"] == count
                    ]
                    for r in ("cublas", "adaptive")
                }
                summary = dict(prompt_length=length, concurrency=count)
                for route, rows in groups.items():
                    wall = sum(r["wall_seconds"] for r in rows)
                    summary[route] = dict(
                        tps=sum(r["output_tokens"] for r in rows) / wall,
                        mean_cohort_seconds=wall / len(rows),
                        mean_server_cpu_core_seconds=statistics.mean(
                            r["server_cpu_core_seconds"] for r in rows
                        ),
                        mean_request_latency_seconds=statistics.mean(
                            q["latency_seconds"] for r in rows for q in r["requests"]
                        ),
                        all_tokens_equal_baseline=all(
                            r["tokens_equal_first_cublas_wave"] for r in rows
                        ),
                        trial_tps=[r["output_tps"] for r in rows],
                    )
                summary["speedup"] = (
                    summary["adaptive"]["tps"] / summary["cublas"]["tps"]
                )
                report["summary"].append(summary)
            report["completed"] = True
            phase("complete")
        except BaseException as error:
            report["error"] = repr(error)
            if isinstance(error, subprocess.CalledProcessError):
                report["driver_stderr"] = error.stderr
            save()
            raise
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            if admission is not None:
                admission.close()
            if batch is not None:
                batch.close()
            if model is not None:
                model.close()
            report["telemetry"] = dict(
                samples=telemetry.samples, errors=telemetry.errors
            )
            save()


if __name__ == "__main__":
    main()
