"""Recompute the rejected graph screen, scopes, telemetry and actual replay work."""
from pathlib import Path
import json
import runpy
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
read = lambda p: json.loads(p.read_text())
telemetry = runpy.run_path(str(OUT.parent / "v100-batched-mtp/summarize.py"))["telemetry"]
base = OUT / "v100/campaign-v5"
campaign = read(base / "campaign-report.json")
result = dict(external_speed_goal_complete=False, screen_ratios=campaign["screen_ratios"], experiments=[], profiles={}, achieved_occupancy=None)
for e in campaign["experiments"]:
    report = read(base / e["label"] / "timed/report.json")
    rows = []
    for w in report["waves"]:
        count = w["request_count"]
        assert w["emitted_tokens"] == 32 * count
        rs = [r["response"] for r in w["requests"]]
        assert all(r["tokens"] == report["case"]["expected_generated_ids"] for r in rs)
        row = dict(requests=count, repeats=1, complete_cohort_output_tps=32*count/w["wall_seconds"], wall_seconds=w["wall_seconds"], server_cpu_core_seconds=w["server_cpu_core_seconds"], mean_client_completion_seconds=statistics.mean(r["client_completion_seconds"] for r in w["requests"]), max_client_completion_seconds=max(r["client_completion_seconds"] for r in w["requests"]), mean_native_ttft_seconds=statistics.mean(r["native_timings"]["native_time_to_first_token_seconds"] for r in rs), mean_native_generation_seconds=statistics.mean(r["native_timings"]["native_generation_seconds"] for r in rs), acceptance=[dict(accepted=r["mtp_stats"]["accepted_tokens"],proposed=r["mtp_stats"]["proposed_tokens"],replayed=r["mtp_stats"]["replayed_input_tokens"]) for r in rs], telemetry={str(g):telemetry(base/e["label"],report,count,g,campaign["utc_offset_seconds"]) for g in [1,0]})
        if count == 1:
            row["post_first_tps"] = 31/(row["mean_native_generation_seconds"]-row["mean_native_ttft_seconds"])
        deltas = []
        for r in rs:
            graph = r["mtp_stats"].get("target_layer_graph")
            if graph is None:
                continue
            warm = read(base/e["label"]/"timed"/f'response-c{count}-r-1-slot{r["id_slot"]}.json')["mtp_stats"]["target_layer_graph"]
            d = {k:graph[k]-warm[k] for k in ["captures","replays","fallback_calls","eager_positioned_warm_calls"]}
            assert d["captures"] == d["eager_positioned_warm_calls"] == 0
            assert d["replays"] == r["mtp_stats"]["verification_calls_by_input_width"]["3"] == 13
            deltas.append(dict(slot=r["id_slot"],**d,nodes=graph["nodes"],device_position_bytes=graph["device_position_bytes"],pinned_position_bytes=graph["pinned_position_bytes"],value_workspace_bytes=graph["value_workspace_bytes"]))
        row["measured_call_graph_deltas"] = deltas
        rows.append(row)
    result["experiments"].append(dict(label=e["label"],library_sha256=e["library_sha256"],rows=rows))
for name in ["mtp2-eager-c1","mtp2-graph-c1","mtp2-graph-c8"]:
    result["profiles"][name] = read(OUT/"v100/final-v6"/(name+"-attribution.json"))
result["trace_correlation_audit"] = read(OUT/"trace-correlation-audit.json")
(OUT/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
for e in result["experiments"]:
    print(e["label"],[(r["requests"],round(r["complete_cohort_output_tps"],3),r.get("post_first_tps"),r["telemetry"]) for r in e["rows"]])
