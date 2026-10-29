"""Common loopback HTTP benchmark for fixed-work pretokenized greedy requests.

One common barrier/global cohort wall. Client end-to-end rate is distinct from
engine-internal reported timing. No concurrent decode rate from summed latencies.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request, urlopen


def post(url, payload):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def process_cpu(pid):
    # utime/stime include all process threads; do not sum submitting-thread times.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def summarize(waves):
    output = []
    for count in sorted({w["request_count"] for w in waves}):
        rows = [w for w in waves if w["request_count"] == count]
        emitted = {w["emitted_tokens"] for w in rows}
        if len(emitted) != 1:
            raise ValueError("Fixed output work differs between repeats")
        wall = statistics.mean(w["wall_seconds"] for w in rows)
        row = dict(request_count=count,trials=len(rows),mean_cohort_wall_seconds=wall,aggregate_complete_request_output_tps=next(iter(emitted))/wall,mean_server_cpu_core_seconds=statistics.mean(w["server_cpu_core_seconds"] for w in rows),mean_client_completion_seconds=statistics.mean(q["client_completion_seconds"] for w in rows for q in w["requests"]),max_client_completion_seconds=max(q["client_completion_seconds"] for w in rows for q in w["requests"]))
        if count == 1:
            responses = [w["requests"][0]["response"] for w in rows]
            if all(isinstance(r.get("native_timings"), dict) for r in responses):
                times = [r["native_timings"] for r in responses]
                row["native_single_post_first_tps"] = (next(iter(emitted))-1)/statistics.mean(t["native_generation_seconds"]-t["native_time_to_first_token_seconds"] for t in times)
            # llama.cpp timings follow its own accounting, not a client TTFT.
            row["engine_reported_timings"] = [r.get("timings", r.get("native_timings")) for r in responses]
        responses = [q["response"] for w in rows for q in w["requests"]]
        if all(isinstance(r.get("scheduler_timings"), dict) for r in responses):
            times = [r["scheduler_timings"] for r in responses]
            row["mean_scheduler_token_availability_ttft_seconds"] = statistics.mean(t["time_to_first_token_seconds"] for t in times)
            row["max_scheduler_inter_token_seconds"] = max(t["max_inter_token_seconds"] for t in times)
            row["scheduler_timing_scope"] = "Per-request submission to native token availability; includes queue/scheduling, not HTTP streaming/native-only timing"
        output.append(row)
    return output


def validate_response(result, prompt, expected, enforce_pinned):
    if result.get("tokens_predicted") != len(expected) or len(result.get("tokens", [])) != len(expected):
        raise AssertionError("Incomplete fixed output work or missing returned token IDs")
    # llama.cpp tokens_cached describes retained KV, not necessarily reused input.
    # prompt_n is the number actually evaluated in this request.
    processed = result.get("timings", {}).get("prompt_n", result.get("tokens_evaluated"))
    if result.get("tokens_evaluated") != len(prompt) or processed != len(prompt):
        raise AssertionError("Prompt IDs/count changed or prompt work was skipped")
    if enforce_pinned and result["tokens"] != expected:
        raise AssertionError("Venturi output no longer matches the pinned native IDs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--server-pid", required=True, type=int)
    p.add_argument("--case", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--concurrency", default="1,8")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--expect-pinned-ids", action="store_true")
    args = p.parse_args()
    counts = [int(n) for n in args.concurrency.split(",")]
    if any(c not in (1,2,4,8) for c in counts) or args.repeats < 1:
        p.error("Use counts 1,2,4,8 and repeats >=1")
    case = json.loads(Path(args.case).read_text())
    prompt, expected = case["prompt_token_ids"], case["expected_generated_ids"]
    out = Path(args.output_directory)
    out.mkdir(exist_ok=False, parents=True)
    report = dict(scope=__doc__,case=case,url=args.url,server_pid=args.server_pid,waves=[],completed=False)

    def save():
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

    save()
    for count in counts:
        with ThreadPoolExecutor(max_workers=count) as pool:
            for repeat in range(-1,args.repeats):
                start = {}

                def begin():
                    start.update(wall=time.perf_counter(),unix=time.time(),cpu=process_cpu(args.server_pid))

                barrier = threading.Barrier(count+1, action=begin, timeout=180)

                def run(slot):
                    body = dict(prompt=prompt,n_predict=len(expected),id_slot=slot,cache_prompt=False,ignore_eos=True,stream=False,return_tokens=True,temperature=0,top_k=1,top_p=1,min_p=0,repeat_penalty=1,seed=42)
                    barrier.wait()
                    result = post(args.url+"/completion",body)
                    latency = time.perf_counter()-start["wall"]
                    # Preserve failing/warmup responses as well as successful timing rows.
                    (out / f"response-c{count}-r{repeat}-slot{slot}.json").write_text(json.dumps(result, indent=2) + "\n")
                    validate_response(result, prompt, expected, args.expect_pinned_ids)
                    return dict(client_completion_seconds=latency,response=result)

                jobs = [pool.submit(run,i) for i in range(count)]
                barrier.wait()
                rows = [j.result(timeout=180) for j in jobs]
                wave = dict(request_count=count,repeat=repeat,wall_seconds=time.perf_counter()-start["wall"],server_cpu_core_seconds=process_cpu(args.server_pid)-start["cpu"],start_unix=start["unix"],end_unix=time.time(),emitted_tokens=count*len(expected),requests=rows)
                if repeat >= 0:
                    report["waves"].append(wave)
                    save()
                    print(json.dumps({k:v for k,v in wave.items() if k != "requests"}),flush=True)
    report["summary"] = summarize(report["waves"])
    report["all_outputs_equal_first_wave"] = all(q["response"]["tokens"] == report["waves"][0]["requests"][0]["response"]["tokens"] for w in report["waves"] for q in w["requests"])
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
