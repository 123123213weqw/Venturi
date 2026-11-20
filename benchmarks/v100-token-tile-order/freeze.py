"""Freeze a rejected ordering experiment without changing earlier capsules."""

from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/v100-token-tile-order"
RAW = OUT / "v100"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


terminal = read(OUT / "terminal-audit.json")
assert len(terminal["workers"]) == 7
assert not terminal["owned_inference_processes"]
assert not terminal["remaining_periodic_nvml_samplers"]
assert not terminal["listening_ports"]
assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
for rel, h in read(OUT / "prior-reference-pins.json").items():
    assert sha(ROOT / rel) == h, rel
prior = read(OUT / "prior-capsule-audit.json")
assert len(prior) == 31
for cap_name in prior:
    cap = ROOT / "benchmarks" / cap_name
    for filename, base in [
        ("source-sha256.json", cap / "source"),
        ("evidence-sha256.json", cap),
    ]:
        p = cap / filename
        if not p.exists():
            continue
        manifest = read(p)
        manifest = manifest.get("files", manifest)
        found, missing = 0, []
        for rel, h in manifest.items():
            p = base / rel
            if not p.exists() and cap_name == "decoder-layer":
                p = base / Path(rel).name
            if p.is_file():
                assert sha(p) == h, str(p)
                found += 1
            else:
                missing.append(rel)
        assert not missing or (cap_name == "decoder-layer" and found == 4)
assert "1 failed, 100 passed" in (RAW / "study-v1/mtp-tests.log").read_text()
assert "104 passed" in (RAW / "study-v4/mtp-tests.log").read_text()
assert "2096 passed, 448 skipped" in (RAW / "study-v4/full-tests.log").read_text()
assert "785 passed, 1759 skipped" in (OUT / "local-tests.log").read_text()
assert "785 passed, 1759 skipped" in (OUT / "restored-local-tests.log").read_text()
build = read(RAW / "study-v1/build-metadata/build.json")
h = build["library_sha256"]
assert h == "ecec6e10052e47eb75fc783f4bfc59e80f8829279037975e784c4294b58a1461"
assert len(build["source_sha256"]) == 9
for snap in terminal["executed_snapshots"].values():
    for rel, v in build["source_sha256"].items():
        assert snap["src/venturi/ops/csrc/" + rel] == v
op = read(RAW / "study-v4/operators/report.json")
assert op["completed"] and len(op["cases"]) == 77 and not op["profiled"]
assert op["candidate_sha256"] == h
for c in op["cases"]:
    assert c["all_output_bytes_equal"] and len(c["waves"]) == 6
    assert all(w["calls"] == 9 for w in c["waves"])
    assert all(v["relative_l2"] < 0.001 for v in c["sampled_fp64"])
qual = read(RAW / "model-v5/qualification/report.json")
assert qual["completed"] and qual["candidate_library_sha256"] == h
assert qual["profiles"]["q8-residual"]["same_profile_bytes_equal"]
q = qual["profiles"]["q8-residual"]["candidate"]
assert len(q["cases"]) == 4 and sum(len(c["all_logit_bytes"]) for c in q["cases"]) == 98
assert (
    len(q["batch"]["private_cache"])
    == len(q["reset_continuation"]["private_cache"])
    == 8
)
q = read(RAW / "model-v5/mtp-batch-qualification/report.json")
assert (
    q["completed"]
    and q["library_sha256"] == h
    and q["selective_reset_untouched_state_exact"]
)
assert len(q["full_references"]) == 4 and len(q["cohorts"]) == 2
assert all(
    c["exact_target_state"]
    and c["exact_corrected_draft_state"]
    and c["native_without_python_math"]
    for c in q["cohorts"]
)
q = read(RAW / "model-v5/mtp2-qualification/report.json")
assert q["completed"] and q["library_sha256"] == h and q["target_recovery"] == "prefix"
assert len(q["cases"]) == 4
assert all(
    c["output_ids_equal"]
    and c["committed_target_state_bytes_equal"]
    and c["committed_draft_state_equal_mtp1"]
    for c in q["cases"]
)
campaign = read(RAW / "campaign-v6/campaign-report.json")
assert campaign["completed"] and campaign["operator_promoted"]
assert not any(campaign["path_promotions"].values())
assert len(campaign["experiments"]) == 6 and not campaign["external_goal_complete"]
for e in campaign["experiments"]:
    assert e["completed"] and e["stable_outputs"] and e["label"].startswith("screen-")
    assert all(s["trials"] == 1 for s in e["summary"])
    r = read(RAW / "campaign-v6" / e["label"] / "timed/report.json")
    assert r["completed"]
    for w in r["waves"]:
        assert w["emitted_tokens"] == 32 * w["request_count"]
        assert all(
            x["response"]["tokens"] == r["case"]["expected_generated_ids"]
            for x in w["requests"]
        )
        if e["mode"] in ["batch", "mtp-batch"]:
            assert len({x["response"]["batch_cohort_id"] for x in w["requests"]}) == 1
            assert {x["response"]["batch_request_count"] for x in w["requests"]} == {
                w["request_count"]
            }
        if e["mtp_depth"]:
            assert all(
                x["response"]["mtp_stats"]["replayed_input_tokens"]
                == x["response"]["mtp_stats"]["forced_proposals"]
                == 0
                for x in w["requests"]
            )
resolved = {}
for snap, files in terminal["executed_snapshots"].items():
    delta = []
    for rel, v in files.items():
        if (ROOT / rel).is_file() and sha(ROOT / rel) == v:
            continue
        p = OUT / "executed-source-deltas" / snap / rel
        assert p.is_file() and sha(p) == v, (snap, rel)
        delta.append(rel)
    resolved[snap] = dict(total=len(files), preserved_executed_delta=delta)
assert campaign["source_sha256"] == terminal["executed_snapshots"]["token-order-src-v2"]
assert (
    read(RAW / "study-v4/executed-source.json")
    == terminal["executed_snapshots"]["token-order-src-v2"]
)
resources = read(RAW / "final-v7/resources.json")
assert resources["library_sha256"] == h
assert (
    resources["achieved_occupancy"] is None
    and not resources["hardware_counter_collection"]
)
for name in ["mtp2-prefix-c1", "ordinary256-c8", "mtp-batch-c8"]:
    m = read(RAW / "final-v7" / (name + "-metadata.json"))
    assert m["profiled"]
    assert (
        read(RAW / "final-v7" / (name + "-attribution.json"))[
            "achieved_hardware_counters"
        ]
        is False
    )
restore = read(OUT / "restored-dispatch-proof.json")
assert (
    restore["model_dispatch_restored"]
    and restore["all_nine_default_native_source_hashes_restored"]
)
control = read(
    ROOT / "benchmarks/v100-batched-mtp/v100/model-v2/build-metadata/build.json"
)
for rel, v in control["source_sha256"].items():
    assert sha(ROOT / "src/venturi/ops/csrc" / rel) == v
assert not read(OUT / "decision.json")["promote_to_model_default"]
assert not read(OUT / "metrics.json")["external_speed_goal_complete"]
assert len(read(OUT / "grid-layout-proof.json")["traces"]) == 6
files = []
for folder in ["src", "tests", "tools", "scripts", "docs"]:
    files.extend(
        p
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
files.extend(ROOT / p for p in ["README.md", "pyproject.toml", "AGENTS.md"])
files.extend(
    OUT / p
    for p in [
        "design.md",
        "summary.md",
        "next-bottlenecks.md",
        "summarize.py",
        "freeze.py",
    ]
)
manifest = {}
for p in sorted(set(files)):
    rel = p.relative_to(ROOT)
    dst = OUT / "source" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    manifest[str(rel)] = sha(p)
(OUT / "source-sha256.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
checks = dict(
    frozen=True,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_files=len(manifest),
    executed_snapshots_verified=resolved,
    library_sha256=h,
    operator_cases=77,
    new_checks=104,
    operator_gate_passed=True,
    trained_target_and_mtp_state_gates_passed=True,
    arithmetic_precision_workspace_and_tp_fences_unchanged=True,
    original_default_dispatch_restored=True,
    three_repeat_engine_matrix_executed=False,
    full_external_speed_goal_complete=False,
    independent_teacher_reference_reused=True,
    broad_bf16_corpus_quality_qualified=False,
    setup_failure_and_aborted_preflights_retained=True,
    all_owned_processes_terminal=True,
    original_gpu_footprint_restored=True,
    achieved_occupancy=None,
    driver_counter_policy_unchanged=True,
    no_local_rust_or_cuda_compilation=True,
    no_commit_or_push=True,
    prior_capsules_unchanged=prior,
)
(OUT / "evidence-check.json").write_text(json.dumps(checks, indent=2) + "\n")
evidence = {
    str(p.relative_to(OUT)): sha(p)
    for p in sorted(OUT.rglob("*"))
    if p.is_file()
    and "source" not in p.relative_to(OUT).parts
    and p.name != "evidence-sha256.json"
    and "__pycache__" not in p.parts
    and p.suffix != ".pyc"
}
(OUT / "evidence-sha256.json").write_text(
    json.dumps(evidence, indent=2, sort_keys=True) + "\n"
)
assert all(sha(OUT / p) == v for p, v in evidence.items())
assert all(sha(OUT / "source" / p) == v for p, v in manifest.items())
print(
    json.dumps(
        dict(
            frozen=True,
            source_files=len(manifest),
            evidence_files=len(evidence),
            prior_capsules_verified=31,
            goal_complete=False,
        )
    )
)
