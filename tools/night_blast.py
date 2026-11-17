"""Night campaign load client: fixed-count concurrent cohorts over the
loopback benchmark adapter. Aggregate TPS divides emitted tokens by the shared
first-request-sent to last-response-received interval (never summed durations).
Greedy, nonstreaming, uncached, ignore_eos work only."""

import argparse
import json
import hashlib
import statistics
import threading
import time
import urllib.request


def qualified_output_ids(report, prompt, n_predict, *, implementation="dequant"):
    """Only an exact self-described qualified workload can seed timing hashes.

    Legacy reports containing just names/output lengths cannot establish input
    identity. Never substitute the last semantic case or a shorter trajectory.
    """
    if report.get("completed") is not True or report.get("passed") is not True:
        raise ValueError("Complete passing model qualification required")
    records = report.get("implementations", {}).get(implementation, {}).get("cases", [])
    matched = [
        r
        for r in records
        if r.get("prompt_token_ids") == list(prompt)
        and r.get("max_new_tokens") == n_predict
    ]
    if len(matched) != 1:
        raise ValueError("Exactly one qualified timing input/budget match required")
    tokens = matched[0].get("generated_ids")
    if (
        not isinstance(tokens, list)
        or len(tokens) != n_predict
        or any(type(x) is not int or x < 0 for x in tokens)
    ):
        raise ValueError("Qualified timing output must be a complete token trajectory")
    return list(tokens)


def blast(port, concurrency, n_predict, prompt_len, vocab, timeout):
    if any(
        type(n) is not int or n <= 0
        for n in (concurrency, n_predict, prompt_len, vocab)
    ):
        raise ValueError("Benchmark dimensions must be positive integers")
    prompt = [(7 * i + 13) % vocab for i in range(prompt_len)]
    results = [None] * concurrency
    start_gate = threading.Barrier(concurrency)

    def worker(idx):
        body = json.dumps(
            {
                "prompt": prompt,
                "n_predict": n_predict,
                "id_slot": idx % 8,
                "ignore_eos": True,
                "stream": False,
                "cache_prompt": False,
                "temperature": 0,
                "repeat_penalty": 1,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        start_gate.wait()
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
            ids = (
                payload.get("tokens", payload.get("output_ids"))
                if isinstance(payload, dict)
                else None
            )
            if (
                not isinstance(ids, list)
                or len(ids) != n_predict
                or any(type(x) is not int or not 0 <= x < vocab for x in ids)
            ):
                payload = {
                    "error": "Incomplete or invalid token response",
                    "response": payload,
                }
            results[idx] = (t0, time.perf_counter(), payload)
        except Exception as exc:  # noqa: BLE001 - record failures verbatim
            results[idx] = (t0, time.perf_counter(), {"error": repr(exc)})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall1 = time.perf_counter()
    errors = [r[2] for r in results if isinstance(r[2], dict) and "error" in r[2]]
    tokens = sum(
        len(r[2].get("tokens", r[2].get("output_ids", [])))
        for r in results
        if isinstance(r[2], dict) and "error" not in r[2]
    )
    interval = wall1 - wall0
    valid = not errors and tokens == concurrency * n_predict
    latencies = [r[1] - r[0] for r in results]
    output_hashes = [
        (
            hashlib.sha256(
                json.dumps(r[2].get("tokens", r[2].get("output_ids", []))).encode()
            ).hexdigest()
            if "error" not in r[2]
            else None
        )
        for r in results
    ]
    return {
        "concurrency": concurrency,
        "n_predict": n_predict,
        "prompt_len": prompt_len,
        "interval_s": round(interval, 4),
        "emitted_tokens": tokens,
        "errors": errors,
        "valid": valid,
        "expected_tokens": concurrency * n_predict,
        "aggregate_tps": round(tokens / interval, 3) if valid else None,
        "partial_success_tps_diagnostic": (
            round(tokens / interval, 3) if not valid else None
        ),
        "response_latency_median_s": statistics.median(latencies),
        "response_latency_max_s": max(latencies),
        "output_sha256": output_hashes,
        "timing_scope": "Common end-to-end wall time including HTTP, admission, prefill and decode; not decode-only or streaming TTFT",
        "native_timings": [r[2].get("native_timings") for r in results],
        "scheduler_timings": [r[2].get("scheduler_timings") for r in results],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=18033)
    p.add_argument("--vocab", type=int, default=248320)
    p.add_argument("--concurrency", type=int, nargs="+", required=True)
    p.add_argument("--n-predict", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--timeout", type=float, default=1200.0)
    p.add_argument("--output")
    a = p.parse_args()
    runs = []
    for count in a.concurrency:
        for repeat in range(a.repeats):
            if a.warmup and repeat == 0:
                blast(a.port, count, a.n_predict, a.prompt_len, a.vocab, a.timeout)
            run = blast(a.port, count, a.n_predict, a.prompt_len, a.vocab, a.timeout)
            runs.append(run)
            print(json.dumps(run), flush=True)
    if a.output:
        with open(a.output, "w") as fh:
            json.dump(runs, fh, indent=1)
    if any(not r["valid"] for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
