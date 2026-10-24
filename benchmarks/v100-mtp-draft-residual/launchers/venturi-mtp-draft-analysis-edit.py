from pathlib import Path
p=Path('benchmarks/v100-native-column-mlp/summarize.py');s=p.read_text().replace('base = OUT / "v100/campaign-v6"','base = OUT / "v100/campaign-v3"')
s=s.replace('profile=e["projection_profile"],','profile=e["projection_profile"],\n                draft_profile=e.get("mtp_draft_projection_profile"),')
s=s.replace('profile=e["profile"],','profile=e["profile"],\n                    draft_profile=e.get("draft_profile"),')
s=s.replace('        if native and external:\n            a = max(native, key=lambda r: r["http_tps"])\n            b = max(external, key=lambda r: r["http_tps"])\n            result["best_repeated_native_http_by_count"][str(n)] = dict(\n                **a, over_fresh_external_ratio=a["http_tps"] / b["http_tps"]\n            )\n            result["fresh_external_http_by_count"][str(n)] = b', '''        if native:
            result["best_repeated_native_http_by_count"][str(n)] = max(native, key=lambda r: r["http_tps"])
        if external:
            result["fresh_external_http_by_count"][str(n)] = max(external, key=lambda r: r["http_tps"])''')
a=s.index('    if "repeat-candidate-mtp2" in ix:');b=s.index('    historical = []',a);s=s[:a]+s[b:]
a=s.index('        native = result["best_repeated_native_http_by_count"]');b=s.index('    (OUT / "metrics.json").write_text',a)
s=s[:a]+'''    previous_capsule = OUT.parent / "v100-native-column-mlp"
    previous = read(previous_capsule / "metrics.json")
    import hashlib
    result["reused_qualified_native_metrics"] = dict(path=str(previous_capsule / "metrics.json"), sha256=hashlib.sha256((previous_capsule / "metrics.json").read_bytes()).hexdigest())
    result["best_known_native_http_by_count"] = {}
    for n in [1,2,4,8]:
        historical_native = dict(**previous["best_repeated_native_http_by_count"][str(n)], source="prior qualified column-MLP campaign", draft_profile="fp16" if previous["best_repeated_native_http_by_count"][str(n)]["depth"] else None)
        candidates = [historical_native]
        fresh = result["best_repeated_native_http_by_count"].get(str(n))
        if fresh:
            fresh["over_best_known_pinned_external_ratio"] = fresh["http_tps"] / result["best_known_pinned_external_http_by_count"][str(n)]["http_tps"]
            candidates.append(dict(**fresh, source="fresh same-library draft campaign"))
        best = max(candidates, key=lambda r:r["http_tps"])
        best["over_best_known_pinned_external_ratio"] = best["http_tps"] / result["best_known_pinned_external_http_by_count"][str(n)]["http_tps"]
        result["best_known_native_http_by_count"][str(n)] = best
    decode_candidates = [dict(native_tps=previous["single_decode_comparison"]["native_tps"], source="prior qualified column-MLP MTP2/FP16 draft", draft_profile="fp16")]
    for e in result["experiments"]:
        if not e["label"].startswith("repeat-"):
            continue
        candidate=e["label"].startswith("repeat-candidate-")
        if candidate and not result["adoption_checks"][e["label"].removeprefix("repeat-candidate-")]["qualifies"]:
            continue
        row=next((r for r in e["rows"] if r["requests"]==1),None)
        if row and "native_post_first_tps" in row:
            decode_candidates.append(dict(native_tps=row["native_post_first_tps"],source=e["label"],draft_profile=e.get("draft_profile")))
    best=max(decode_candidates,key=lambda r:r["native_tps"])
    external=max([r["post_first_tps"] for r in historical if r["requests"]==1]+[previous["single_decode_comparison"]["best_known_pinned_external_tps"]])
    result["single_decode_comparison"]=dict(**best,best_known_pinned_external_tps=external,best_known_goal_floor=external*1.05,over_best_known_ratio=best["native_tps"]/external,additional_gain_to_best_known_goal_ratio=external*1.05/best["native_tps"],best_known_goal_met=best["native_tps"]>=external*1.05)
    result["complete_http_matrix_goal_met"]=all(r["over_best_known_pinned_external_ratio"]>=1.05 for r in result["best_known_native_http_by_count"].values())
    result["external_speed_goal_complete"]=result["complete_http_matrix_goal_met"] and result["single_decode_comparison"]["best_known_goal_met"]
    result["full_goal_completion_still_requires_final_audit"] = True
''' +s[b:]
Path('benchmarks/v100-mtp-draft-residual/summarize.py').write_text(s)
