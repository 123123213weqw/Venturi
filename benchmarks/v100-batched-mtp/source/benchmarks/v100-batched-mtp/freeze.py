from pathlib import Path
import datetime
import hashlib
import json
import shutil

root = Path(__file__).resolve().parents[2]
out = root / "benchmarks/v100-batched-mtp"
raw = out / "v100"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


t = read(out / "terminal-audit.json")
assert (
    len(t["workers"]) == 9
    and not t["owned_inference_processes"]
    and not t["listening_ports"]
    and not t["remaining_periodic_nvml_samplers"]
)
assert "RmProfilingAdminOnly: 1" in t["counter_policy"]
for rel, h in read(out / "prior-reference-pins.json").items():
    assert sha(root / rel) == h, rel
prior = read(out / "prior-capsule-audit.json")
assert len(prior) == 30
for cap_name in prior:
    cap = root / "benchmarks" / cap_name
    for name, base in [
        ("source-sha256.json", cap / "source"),
        ("evidence-sha256.json", cap),
    ]:
        p = cap / name
        if not p.exists():
            continue
        manifest = read(p)
        manifest = manifest.get("files", manifest)
        n = 0
        missing = []
        for rel, h in manifest.items():
            p = base / rel
            if not p.exists() and cap_name == "decoder-layer":
                p = base / Path(rel).name
            if p.is_file():
                assert sha(p) == h, str(p)
                n += 1
            else:
                missing.append(rel)
        if missing:
            assert cap_name == "decoder-layer" and n == 4
assert "1990 passed, 448 skipped" in (raw / "model-v2/full-tests.log").read_text()
assert "1992 passed, 448 skipped" in (raw / "sdk-tests-v7/full-tests.log").read_text()
assert "86 passed" in (raw / "sdk-tests-v7/mtp-tests.log").read_text()
assert (
    "2 failed, 1990 passed, 448 skipped"
    in (raw / "sdk-tests-v5/full-tests.log").read_text()
)
assert "785 passed, 1655 skipped" in (out / "local-sdk-tests.log").read_text()
q = read(raw / "model-v2/qualification/report.json")
assert (
    q["completed"]
    and q["selective_reset_untouched_state_exact"]
    and len(q["full_references"]) == 4
    and len(q["cohorts"]) == 2
)
assert all(
    c["exact_target_state"]
    and c["exact_corrected_draft_state"]
    and c["native_without_python_math"]
    for c in q["cohorts"]
)
b = read(raw / "model-v2/build-metadata/build.json")
h = "f07bf75a975b963911aff5fefd5614dbf0b8b5b74e499f5525b6ad5d5317369a"
assert b["library_sha256"] == q["library_sha256"] == h and len(b["source_sha256"]) == 9
for rel, v in b["source_sha256"].items():
    assert sha(root / "src/venturi/ops/csrc" / rel) == v
for stage in ["campaign-v3", "sdk-campaign-v8"]:
    c = read(raw / stage / "campaign-report.json")
    assert c["completed"] and c["promoted"] and not c["external_goal_complete"]
    for e in c["experiments"]:
        assert e["completed"] and e["stable_outputs"]
        r = read(raw / stage / e["label"] / "timed/report.json")
        assert r["completed"]
        repeats = 1 if e["label"].startswith("screen") else 3
        assert all(v["trials"] == repeats for v in e["summary"])
        for w in r["waves"]:
            assert w["emitted_tokens"] == 32 * w["request_count"] and all(
                x["response"]["tokens"] == r["case"]["expected_generated_ids"]
                for x in w["requests"]
            )
            if e["mode"] in ["batch", "mtp-batch"]:
                assert {
                    x["response"]["batch_request_count"] for x in w["requests"]
                } == {w["request_count"]}
                assert (
                    len({x["response"]["batch_cohort_id"] for x in w["requests"]}) == 1
                )
            if e["mtp_depth"]:
                assert all(
                    x["response"]["mtp_stats"]["forced_proposals"] == 0
                    for x in w["requests"]
                )
                if e["mode"] == "mtp-batch" or e["mtp_recovery"] == "prefix":
                    assert all(
                        x["response"]["mtp_stats"]["replayed_input_tokens"] == 0
                        for x in w["requests"]
                    )

resolved = {}
for snap, files in t["executed_snapshots"].items():
    delta = []
    for rel, v in files.items():
        if (root / rel).is_file() and sha(root / rel) == v:
            continue
        p = out / "executed-source-deltas" / snap / rel
        assert p.is_file() and sha(p) == v, (snap, rel)
        delta.append(rel)
    resolved[snap] = dict(total=len(files), preserved_executed_delta=delta)
assert (
    read(raw / "sdk-campaign-v8/campaign-report.json")["source_sha256"]
    == t["executed_snapshots"]["batched-mtp-src-v5"]
)
assert (
    read(raw / "campaign-v3/campaign-report.json")["source_sha256"]
    == t["executed_snapshots"]["batched-mtp-src-v2"]
)
assert (
    read(raw / "sdk-tests-v7/executed-source.json")
    == t["executed_snapshots"]["batched-mtp-src-v5"]
)
for stage in ["final-v4", "sdk-final-v9"]:
    assert read(raw / stage / "completed.json")["sha256"] == h
    for n in [1, 8]:
        p = read(raw / stage / f"mtp-batch-c{n}-metadata.json")
        assert (
            p["profiled"]
            and p["logical_requests"] == n
            and p["native_call_ranges"] == 1
            and p["library_sha256"] == h
        )
        assert len(p["outputs"]) == n and all(len(row) == 32 for row in p["outputs"])
        assert (
            read(raw / stage / f"mtp-batch-c{n}-attribution.json")[
                "achieved_hardware_counters"
            ]
            is False
        )
resources = read(raw / "final-v4/resources.json")
assert (
    resources["library_sha256"] == h
    and resources["achieved_occupancy"] is None
    and not resources["hardware_counter_collection"]
)
assert not read(out / "metrics.json")["external_speed_goal_complete"]
files = []
for folder in ["src", "tests", "tools", "scripts", "docs"]:
    files.extend(
        p
        for p in (root / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
files.extend(root / p for p in ["README.md", "pyproject.toml", "AGENTS.md"])
files.extend(
    out / p
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
    rel = p.relative_to(root)
    dst = out / "source" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    manifest[str(rel)] = sha(p)
(out / "source-sha256.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
checks = dict(
    frozen=True,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_files=len(manifest),
    executed_snapshots_verified=resolved,
    library_sha256=h,
    native_batched_target_verification=True,
    draft_block_and_prefill_batched=False,
    duplicate_permanent_target_cache=False,
    hot_path_metadata_guard_once_per_public_info_stats_generate=True,
    native_source_guards_unchanged=True,
    existing_arithmetic_profiles_unchanged=True,
    new_cpu_teacher_campaign=False,
    independent_target_quality_reference_reused=True,
    broad_bf16_corpus_quality_qualified=False,
    three_repeat_target_batch_matrix=True,
    three_repeat_sdk_matrix=True,
    stronger_controls_explicitly_reused=True,
    automatic_universal_strategy=False,
    external_speed_goal_complete=False,
    failed_snapshot_and_preflight_preserved=True,
    all_owned_processes_terminal=True,
    original_gpu_footprint_restored=True,
    achieved_occupancy=None,
    driver_counter_policy_unchanged=True,
    no_commit_or_push=True,
    no_local_rust_or_cuda_compilation=True,
    prior_capsules_unchanged=prior,
)
(out / "evidence-check.json").write_text(json.dumps(checks, indent=2) + "\n")
evidence = {
    str(p.relative_to(out)): sha(p)
    for p in sorted(out.rglob("*"))
    if p.is_file()
    and "source" not in p.relative_to(out).parts
    and p.name != "evidence-sha256.json"
    and "__pycache__" not in p.parts
    and p.suffix != ".pyc"
}
(out / "evidence-sha256.json").write_text(
    json.dumps(evidence, indent=2, sort_keys=True) + "\n"
)
assert all(sha(out / p) == v for p, v in evidence.items())
assert all(sha(out / "source" / p) == v for p, v in manifest.items())
print(
    json.dumps(
        dict(
            frozen=True,
            source_files=len(manifest),
            evidence_files=len(evidence),
            prior_capsules_verified=30,
            goal_complete=False,
        )
    )
)
