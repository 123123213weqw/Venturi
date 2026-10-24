"""Loopback-only completion adapter for matched benchmarks, not a production server.

Python owns HTTP and preallocated slot submission only; one native generate call
owns all model/token execution. Fixed greedy, no cache reuse, no streaming.
"""

import argparse
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
    if not isinstance(ids, list) or not ids or len(ids) > 32 or any(type(t) is not int or not 0 <= t < vocab for t in ids):
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--mtp-artifact")
    p.add_argument("--mode", choices=("tp", "pp"), required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--ready-file", required=True)
    args = p.parse_args()
    if not 1 <= args.slots <= 8 or args.capacity < 55:
        p.error("Use 1..8 slots with capacity >= 55")
    import torch
    from venturi.models import NativeQwenModel, NativeQwenPipelineModel, NativeQwenMTPModel, QwenQ8Weights, QwenMTPWeights
    from venturi.ninfer import NInferArtifact
    from venturi.ops import CudaBackend

    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    backend = CudaBackend(args.library)
    kw = dict(weight_mode="resident", dtype=torch.float16, device="cuda:0", max_tokens=32)
    target = (
        NativeQwenModel.from_q8(weights, backend, tensor_parallel_device="cuda:1", tensor_parallel_transport="host", **kw)
        if args.mode == "tp" else
        NativeQwenPipelineModel.from_q8(weights, backend, pipeline_device="cuda:1", split_layer=32, transport="host", **kw)
    )
    model = target
    sessions = []
    try:
        if args.mtp_artifact:
            model = NativeQwenMTPModel.from_q8(target, QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact)))
        sessions = [model.create_session(args.capacity) for _ in range(args.slots)]
        locks = [threading.Lock() for _ in sessions]
        metadata = dict(engine="Venturi native",mode=args.mode,slots=args.slots,capacity=args.capacity,mtp_depth=1 if args.mtp_artifact else 0,scope=__doc__)

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
                    ids, count, slot = request_values(json.loads(self.rfile.read(size)), args.slots, args.capacity, weights.config["vocab_size"])
                    with locks[slot]:
                        session = sessions[slot]
                        session.reset()
                        output = session.generate(ids, max_new_tokens=count, eos_token_ids=[])
                        timings = session.generation_timings()
                        result = dict(tokens=output,tokens_predicted=len(output),tokens_evaluated=len(ids),tokens_cached=0,id_slot=slot,native_timings=timings,mtp_stats=session.stats() if args.mtp_artifact else None)
                    self.reply(200, result)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self.reply(400, dict(error=str(error)))
                except Exception as error:
                    self.reply(500, dict(error=str(error)))

        def stop(*unused):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        Path(args.ready_file).write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        for session in sessions:
            session.close()
        if model is not target:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
