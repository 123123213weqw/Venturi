"""Freeze paired-plane ablations, trained gates and serving evidence."""

from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/v100-activation-words"
RAW = OUT / "v100"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


terminal = read(OUT / "terminal-audit.json")
assert len(terminal["workers"]) == 10
assert not terminal["owned_inference_processes"]
assert not terminal["remaining_periodic_nvml_samplers"]
assert not terminal["listening_ports"]
assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
for rel, h in read(OUT / "prior-reference-pins.json").items():
    assert sha(ROOT / rel) == h, rel
prior = read(OUT / "prior-capsule-audit.json")
assert len(prior) == 32
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
assert "786 passed, 1851 skipped" in (OUT / "local-tests.log").read_text()
assert "786 passed, 1851 skipped" in (OUT / "guarded-local-tests.log").read_text()
for variant in ["study-v1", "bound-study-v3", "aos-five-study-v5", "guarded-study-v7"]:
    folder = RAW / variant
    assert "92 passed" in (folder / "mtp-tests.log").read_text()
    assert "2085 passed, 552 skipped" in (folder / "full-tests.log").read_text()
    so = read(folder / "completed.json")["sha256"]
    build = next((folder / "build").rglob("build.json"))
    build = read(build)
    assert build["library_sha256"] == so and len(build["source_sha256"]) == 9
    op = read(folder / "operators/report.json")
    assert op["completed"] and len(op["cases"]) == 33 and not op["profiled"]
    assert op["candidate_sha256"] == so
    for cell in op["cases"]:
        assert cell["all_output_bytes_equal"]
        assert all(w["calls"] == 9 for w in cell["waves"])
        assert len(cell["waves"]) in [6, 9, 12]
        assert all(v["relative_l2"] < 0.001 for v in cell["sampled_fp64"])
    snap = {
        "study-v1": "v1",
        "bound-study-v3": "v2",
        "aos-five-study-v5": "v3",
        "guarded-study-v7": "v4",
    }[variant]
    executed = terminal["executed_snapshots"]["activation-words-src-" + snap]
    for rel, h0 in build["source_sha256"].items():
        assert executed["src/venturi/ops/csrc/" + rel] == h0
h = read(RAW / "guarded-study-v7/completed.json")["sha256"]
campaign = read(RAW / "guarded-campaign-v9/campaign-report.json")
assert campaign["completed"] and not campaign["external_goal_complete"]
promoted = campaign["operator_promoted"]
if promoted:
    qual = read(RAW / "guarded-model-v8/qualification/report.json")
    assert qual["completed"] and qual["candidate_library_sha256"] == h
    assert qual["profiles"]["q8-residual"]["same_profile_bytes_equal"]
    q = qual["profiles"]["q8-residual"]["candidate"]
    assert (
        len(q["cases"]) == 4
        and sum(len(c["all_logit_bytes"]) for c in q["cases"]) == 98
    )
    assert (
        len(q["batch"]["private_cache"])
        == len(q["reset_continuation"]["private_cache"])
        == 8
    )
    q = read(RAW / "guarded-model-v8/mtp-batch-qualification/report.json")
    assert (
        q["completed"]
        and q["library_sha256"] == h
        and q["selective_reset_untouched_state_exact"]
    )
    assert all(
        c["exact_target_state"]
        and c["exact_corrected_draft_state"]
        and c["native_without_python_math"]
        for c in q["cohorts"]
    )
    q = read(RAW / "guarded-model-v8/mtp2-qualification/report.json")
    assert (
        q["completed"] and q["library_sha256"] == h and q["target_recovery"] == "prefix"
    )
    assert len(q["cases"]) == 4 and all(
        c["output_ids_equal"]
        and c["committed_target_state_bytes_equal"]
        and c["committed_draft_state_equal_mtp1"]
        for c in q["cases"]
    )
for e in campaign["experiments"]:
    assert e["completed"] and e["stable_outputs"]
    repeated = e["label"].startswith("repeat-")
    assert all(s["trials"] == (3 if repeated else 1) for s in e["summary"])
    r = read(RAW / "guarded-campaign-v9" / e["label"] / "timed/report.json")
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
assert (
    campaign["source_sha256"]
    == terminal["executed_snapshots"]["activation-words-src-v4"]
)
resources = read(RAW / "guarded-final-v10/resources.json")
assert (
    resources["library_sha256"] == h
    and resources["achieved_occupancy"] is None
    and not resources["hardware_counter_collection"]
)
if promoted:
    for name in ["mtp2-prefix-c1", "ordinary256-c8", "mtp-batch-c8"]:
        assert read(RAW / "guarded-final-v10" / (name + "-metadata.json"))["profiled"]
        assert (
            read(RAW / "guarded-final-v10" / (name + "-attribution.json"))[
                "achieved_hardware_counters"
            ]
            is False
        )
assert not read(OUT / "metrics.json")["external_speed_goal_complete"]
decision = read(OUT / "decision.json")
control = read(
    ROOT / "benchmarks/v100-batched-mtp/v100/model-v2/build-metadata/build.json"
)
selected = (
    read(next((RAW / "guarded-study-v7/build").rglob("build.json")))
    if decision["promote_to_model_default"]
    else control
)
for rel, v in selected["source_sha256"].items():
    assert sha(ROOT / "src/venturi/ops/csrc" / rel) == v
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
        "decide.py",
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
    operator_cases_per_variant=33,
    operator_variants=4,
    checks_per_variant=92,
    operator_gate_passed=promoted,
    trained_target_and_mtp_state_gates_passed=promoted,
    arithmetic_precision_workspace_and_tp_fences_unchanged=True,
    original_default_dispatch_restored=not decision["promote_to_model_default"],
    guarded_native_dispatch_adopted=decision["promote_to_model_default"],
    three_repeat_engine_matrix_executed=any(campaign["path_promotions"].values()),
    full_external_speed_goal_complete=False,
    independent_teacher_reference_reused=True,
    broad_bf16_corpus_quality_qualified=False,
    initial_failed_operator_gate_retained=True,
    three_descriptor_regressions_retained=True,
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
            prior_capsules_verified=32,
            goal_complete=False,
        )
    )
)
