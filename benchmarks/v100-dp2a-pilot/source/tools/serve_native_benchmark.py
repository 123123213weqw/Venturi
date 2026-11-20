"""Loopback-only completion adapter for matched benchmarks, not a production server.

Python owns HTTP and preallocated slot submission only; one native generate call
owns all model/token execution. Fixed greedy, no cache reuse, no streaming.
"""

import argparse
from concurrent.futures import Future
from queue import Queue, Empty
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading


def request_values(body, slots, capacity, vocab):
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object")
    ids = body.get("prompt")
    count, slot = body.get("n_predict"), body.get("id_slot")
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) > 32
        or any(type(t) is not int or not 0 <= t < vocab for t in ids)
    ):
        raise ValueError("Expected nonempty pretokenized prompt")
    if type(count) is not int or not 1 <= count <= 128 or len(ids) + count > capacity:
        raise ValueError("Output budget exceeds benchmark slot capacity")
    if type(slot) is not int or not 0 <= slot < slots:
        raise ValueError("Explicit valid slot required")
    if body.get("stream") or body.get("cache_prompt") or not body.get("ignore_eos"):
        raise ValueError("Benchmark requires nonstreaming, uncached, fixed output work")
    if body.get("temperature") != 0 or body.get("repeat_penalty", 1) != 1:
        raise ValueError("Benchmark supports unpenalized greedy only")
    return ids, count, slot


class CompletionBatcher:
    """Bounded cohort admission only; all token execution is one native call."""

    def __init__(self, batch, slots, window_seconds):
        self.batch, self.slots, self.window = batch, slots, window_seconds
        self.queue, self.lock, self.closed = Queue(), threading.Lock(), False
        self.cohort = 0
        self.thread = threading.Thread(
            target=self._run, name="native-batch-admission", daemon=False
        )
        self.thread.start()

    def submit(self, slot, prompt, maximum):
        with self.lock:
            if self.closed:
                raise RuntimeError("Batch admission is closed")
            future = Future()
            self.queue.put((slot, prompt, maximum, future, time.perf_counter()))
            return future

    def _run(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            tasks = [first]
            end = time.perf_counter() + self.window
            closing = False
            while len(tasks) < self.slots:
                remaining = end - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    next_task = self.queue.get(timeout=remaining)
                except Empty:
                    break
                if next_task is None:
                    closing = True
                    break
                tasks.append(next_task)
            tasks = [t for t in tasks if not t[3].cancelled()]
            if tasks:
                try:
                    began = time.perf_counter()
                    selected = [t[0] for t in tasks]
                    self.batch.reset(slots=selected)
                    outputs = self.batch.generate(
                        [t[1] for t in tasks],
                        max_new_tokens=[t[2] for t in tasks],
                        eos_token_ids=[],
                        slots=selected,
                    )
                    info = self.batch.info()
                    self.cohort += 1
                    stats = {
                        k: info[k]
                        for k in (
                            "group_forward_calls",
                            "input_rows",
                            "decode_group_calls",
                            "prefill_group_calls",
                            "head_calls",
                            "head_rows",
                            "max_group_rows",
                        )
                    }
                    for task, output in zip(tasks, outputs):
                        slot, prompt, _, future, enqueued = task
                        if future.done():
                            continue
                        timing = info["requests"][slot]
                        future.set_result(
                            dict(
                                tokens=output,
                                tokens_predicted=len(output),
                                tokens_evaluated=len(prompt),
                                tokens_cached=0,
                                id_slot=slot,
                                native_timings={
                                    k: timing[k]
                                    for k in (
                                        "native_time_to_first_token_seconds",
                                        "native_generation_seconds",
                                    )
                                },
                                batch_cohort_id=self.cohort,
                                batch_request_count=len(tasks),
                                batch_admission_delay_seconds=began - enqueued,
                                batch_stats=stats,
                                batch_stats_scope="Shared per cohort, do not sum across responses",
                                **(
                                    {
                                        "mtp_stats": timing["mtp"],
                                        "mtp_batch_stats": {
                                            k: info[k]
                                            for k in (
                                                "shared_verification_wall_seconds",
                                                "shared_prefill_complete_seconds",
                                                "shared_native_generation_seconds",
                                                "mtp_model_device_bytes",
                                                "all_mtp_request_device_bytes",
                                                "all_target_backup_device_bytes",
                                                "shared_mtp_selection_device_bytes",
                                                "prefill_strategy",
                                                "draft_strategy",
                                                "target_verification",
                                            )
                                        },
                                        "mtp_batch_stats_scope": "Shared per cohort; count once, not per response",
                                    }
                                    if "mtp" in timing
                                    else {}
                                ),
                            )
                        )
                except Exception as error:
                    for task in tasks:
                        if not task[3].done():
                            task[3].set_exception(error)
            if closing:
                return

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.queue.put(None)
        self.thread.join(timeout=180)
        if self.thread.is_alive():
            raise RuntimeError("Native batch admission did not settle")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument(
        "--projection-rows",
        type=int,
        choices=(32, 64, 128, 256),
        default=32,
        help="Internal projection workspace rows; not per-slot context capacity",
    )
    p.add_argument("--mtp-artifact")
    p.add_argument("--target-layer-graph", action="store_true")
    p.add_argument("--mtp-depth", type=int, choices=(1, 2), default=1)
    p.add_argument("--mtp-recovery", choices=("replay", "prefix"), default="replay")
    p.add_argument("--mode", choices=("tp", "pp"), required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--ready-file", required=True)
    p.add_argument("--batch-target", action="store_true")
    p.add_argument(
        "--batch-mtp",
        action="store_true",
        help="Native MTP1 target-verification cohorts; serial native drafting/prefill",
    )
    p.add_argument(
        "--projection-profile",
        choices=("fp16", "q8-residual", "q8-residual-column-mlp"),
        default="fp16",
    )
    p.add_argument("--batch-window-ms", type=float, default=5.0)
    args = p.parse_args()
    if args.target_layer_graph and (
        args.mode != "tp"
        or not args.mtp_artifact
        or args.mtp_depth != 2
        or args.mtp_recovery != "prefix"
        or args.projection_profile != "q8-residual"
        or args.batch_target
        or args.batch_mtp
    ):
        p.error(
            "Target layer graphs require scalar resident TP residual MTP2 prefix mode"
        )
    if not 1 <= args.slots <= 8 or args.capacity < 55:
        p.error("Use 1..8 slots with capacity >= 55")
    if args.batch_mtp and (
        args.batch_target
        or args.mode != "tp"
        or not args.mtp_artifact
        or args.mtp_depth != 1
        or args.mtp_recovery != "replay"
    ):
        p.error(
            "Batch MTP requires TP, an MTP artifact, depth one and no scalar recovery/batch-target override"
        )
    if args.mtp_recovery != "replay" and (not args.mtp_artifact or args.mtp_depth != 2):
        p.error("Prefix recovery requires MTP depth two")
    if args.mtp_depth != 1 and not args.mtp_artifact:
        p.error("--mtp-depth requires --mtp-artifact")
    if args.batch_target and (args.mode != "tp" or args.mtp_artifact):
        p.error("Grouped target batching currently requires TP without MTP")
    if args.projection_profile != "fp16" and args.mode != "tp":
        p.error("Explicit residual profile currently supports TP, not PP")
    if not 0 <= args.batch_window_ms <= 100:
        p.error("Batch admission window must be 0..100 ms")
    import torch
    from venturi.models import (
        NativeQwenModel,
        NativeQwenPipelineModel,
        NativeQwenMTPModel,
        QwenQ8Weights,
        QwenMTPWeights,
    )
    from venturi.ninfer import NInferArtifact
    from venturi.ops import CudaBackend

    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    backend = CudaBackend(args.library)
    kw = dict(
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=args.projection_rows,
    )
    target = (
        NativeQwenModel.from_q8(
            weights,
            backend,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            projection_profile=args.projection_profile,
            **kw,
        )
        if args.mode == "tp"
        else NativeQwenPipelineModel.from_q8(
            weights,
            backend,
            pipeline_device="cuda:1",
            split_layer=32,
            transport="host",
            **kw,
        )
    )
    model = target
    sessions = []
    batch = admission = None
    try:
        if args.mtp_artifact:
            model = NativeQwenMTPModel.from_q8(
                target, QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact))
            )
        if args.batch_target or args.batch_mtp:
            batch = model.create_batch([args.capacity] * args.slots)
            admission = CompletionBatcher(
                batch, args.slots, args.batch_window_ms / 1000
            )
        else:
            sessions = [model.create_session(args.capacity) for _ in range(args.slots)]
        if args.mtp_artifact and args.mtp_recovery == "prefix":
            for session in sessions:
                session.prepare_prefix()
        if args.target_layer_graph:
            for session in sessions:
                session.enable_target_layer_graph()
        locks = [threading.Lock() for _ in range(args.slots)]
        metadata = dict(
            target_layer_graph=args.target_layer_graph,
            engine="Venturi native",
            mode=args.mode,
            slots=args.slots,
            capacity=args.capacity,
            projection_rows=args.projection_rows,
            mtp_depth=args.mtp_depth if args.mtp_artifact else 0,
            mtp_recovery="request-prefix1-trace"
            if args.batch_mtp
            else args.mtp_recovery
            if args.mtp_artifact
            else None,
            scope=__doc__,
            batched_target=args.batch_target or args.batch_mtp,
            batched_mtp=args.batch_mtp,
            batch_window_ms=args.batch_window_ms
            if args.batch_target or args.batch_mtp
            else None,
            projection_profile=args.projection_profile,
            mtp_draft_projection_profile="fp16" if args.mtp_artifact else None,
        )

        class Handler(BaseHTTPRequestHandler):
            def reply(self, code, value):
                payload = json.dumps(value, allow_nan=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

            def do_GET(self):
                self.reply(200 if self.path == "/health" else 404, metadata)

            def do_POST(self):
                if self.path != "/completion":
                    self.reply(404, dict(error="Unknown route"))
                    return
                try:
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 1024 * 1024:
                        raise ValueError("Invalid request size")
                    ids, count, slot = request_values(
                        json.loads(self.rfile.read(size)),
                        args.slots,
                        args.capacity,
                        weights.config["vocab_size"],
                    )
                    with locks[slot]:
                        if admission is not None:
                            result = admission.submit(slot, ids, count).result(
                                timeout=180
                            )
                        else:
                            session = sessions[slot]
                            session.reset()
                            output = session.generate(
                                ids,
                                max_new_tokens=count,
                                eos_token_ids=[],
                                **(
                                    {
                                        "draft_tokens": args.mtp_depth,
                                        "recovery": args.mtp_recovery,
                                    }
                                    if args.mtp_artifact
                                    else {}
                                ),
                            )
                            timings = session.generation_timings()
                            result = dict(
                                tokens=output,
                                tokens_predicted=len(output),
                                tokens_evaluated=len(ids),
                                tokens_cached=0,
                                id_slot=slot,
                                native_timings=timings,
                                mtp_stats=session.stats()
                                if args.mtp_artifact
                                else None,
                            )
                    self.reply(200, result)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self.reply(400, dict(error=str(error)))
                except Exception as error:
                    self.reply(500, dict(error=str(error)))

        def stop(*unused):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)

        class Server(ThreadingHTTPServer):
            request_queue_size = 128

        server = Server(("127.0.0.1", args.port), Handler)
        Path(args.ready_file).write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        if admission is not None:
            admission.close()
        if batch is not None:
            batch.close()
        for session in sessions:
            session.close()
        if model is not target:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
