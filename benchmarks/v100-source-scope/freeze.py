"""Freeze the measured execution-policy experiment, including its failed check."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / "v100"
LIBRARIES = {
    "0": "061290e9ccd089017161ca9265348526bf7f71d1db1c9abc5c8fdec91c2d677b",
    "1": "1b82d19104f4987db96672e74430d3cf0a6da51997c951a4e54ecd57fcef18b9",
}


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT / "evidence-sha256.json").exists(), "Never replace frozen evidence"
    prior = read(OUT / "prior-capsule-audit.json")
    assert len(prior) == 40
    for rel, digest in read(OUT / "prior-reference-pins.json").items():
        assert sha(ROOT / rel) == digest, rel
    verified = {}
    for name in prior:
        cap = ROOT / "benchmarks" / name
        count, missing = 0, []
        for filename, base in [
            ("source-sha256.json", cap / "source"),
            ("evidence-sha256.json", cap),
        ]:
            if not (cap / filename).exists():
                continue
            manifest = read(cap / filename)
            manifest = manifest.get("files", manifest)
            found = 0
            for rel, digest in manifest.items():
                p = base / rel
                if not p.is_file() and name == "decoder-layer":
                    p = base / Path(rel).name
                if p.is_file():
                    assert sha(p) == digest, str(p)
                    count += 1
                    found += 1
                else:
                    missing.append(rel)
            assert not missing or (name == "decoder-layer" and found == 4)
        verified[name] = dict(verified_files=count, legacy_missing=missing)
    terminal = read(OUT / "terminal-audit.json")
    assert {k: v["exit_code"] for k, v in terminal["workers"].items()} == {
        "fixtures-v1": 1,
        "fixtures-v2": 0,
        "model-v3": 0,
        "campaign-v4": 0,
        "profile-v5": 0,
    }
    assert (
        not terminal["owned_processes"]
        and not terminal["remaining_periodic_samplers"]
        and not terminal["port18953"]
    )
    assert "3651 MiB" in terminal["gpu_after"] and "3189 MiB" in terminal["gpu_after"]
    assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
    restoration = read(OUT / "production-restoration.json")
    assert (
        restoration["restored"] and restoration["no_unrelated_user_changes_overwritten"]
    )
    assert (
        len(restoration["restored_files"]) == 11
        and len(restoration["removed_prototype_only_files"]) == 3
    )
    prior_production = read(
        ROOT / "benchmarks/v100-source-validation/source-manifest-v1.json"
    )
    assert restoration["restored_source_manifest"] == prior_production
    current_production = {
        str(p.relative_to(ROOT)): sha(p)
        for d in ["src", "tests", "tools"]
        for p in (ROOT / d).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    assert current_production == prior_production
    experiment = OUT / "executed-source-v4"
    for n in range(1, 5):
        manifest = read(OUT / f"source-manifest-v{n}.json")
        assert terminal["source_manifests"][str(n)] == manifest
        assert all(
            sha(OUT / f"executed-source-v{n}" / rel) == h for rel, h in manifest.items()
        )
        # All inference sources stayed fixed after the first remote compilation.
        assert all(
            sha(experiment / rel) == h
            for rel, h in manifest.items()
            if rel.startswith("src/")
        )
    assert all(
        sha(experiment / rel) == h
        for rel, h in read(OUT / "source-manifest-v4.json").items()
    )
    builds = [read(p) for p in (RAW / "fixtures-v1/build").rglob("build.json")]
    assert len(builds) == 2
    for build in builds:
        scope = str(int(build["resident_source_scope"]))
        assert build["library_sha256"] == LIBRARIES[scope]
        assert len(build["source_sha256"]) == 10
        assert all(
            sha(experiment / "src/venturi/ops/csrc" / rel) == h
            for rel, h in build["source_sha256"].items()
        )
        assert ("-DVENTURI_RESIDENT_SOURCE_SCOPE=1" in build["command"]) == bool(
            int(scope)
        )
    first = (RAW / "fixtures-v1/focused-tests.log").read_text()
    assert "1 failed, 19 passed" in first and "Native Qwen source changed" in first
    assert "29 passed" in (RAW / "fixtures-v2/focused-tests.log").read_text()
    assert (
        "2269 passed, 687 skipped" in (RAW / "fixtures-v2/full-tests.log").read_text()
    )
    assert "835 passed, 2122 skipped" in (OUT / "local-tests-v4.log").read_text()
    assert "822 passed, 2105 skipped" in (OUT / "local-restored-tests.log").read_text()
    accepted_build = read(
        ROOT
        / "benchmarks/v100-mtp-draft-residual/v100/fixtures-v1/build/089e914d53d145f0/build.json"
    )
    assert (
        accepted_build["library_sha256"]
        == "6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2"
    )
    assert all(
        sha(ROOT / "src/venturi/ops/csrc" / p) == h
        for p, h in accepted_build["source_sha256"].items()
    )
    assert "5 passed" in (RAW / "profile-v5/parser-tests.log").read_text()
    assert "All checks passed!" in (OUT / "ruff.log").read_text()
    quality = read(RAW / "model-v3/qualification/report.json")
    assert quality["completed"] and quality["source_scope_mode"] == 1
    assert quality["library_sha256"] == LIBRARIES["1"]
    assert (
        quality["independent_numerical_quality_reused"]
        and not quality["new_cpu_teacher_campaign"]
    )
    assert len(quality["cases"]) == 4 and all(
        c["exact_ordinary_target_and_pinned_state"]
        and c["exact_pinned_corrected_draft_state"]
        and c["exact_prior_mtp2_algorithm_counters"]
        and c["native_without_python_math"]
        for c in quality["cases"]
    )
    pins = read(OUT / "reused-quality-reference-pins.json")
    assert quality["reused_reference_sha256"] == {
        k: v["sha256"] for k, v in pins.items()
    }
    batch = read(RAW / "model-v3/batch-qualification/report.json")
    old = read(
        ROOT
        / "benchmarks/v100-mtp-draft-residual/v100/model-v2/batch-qualification/report.json"
    )
    assert batch["completed"] and batch["selective_reset_untouched_state_exact"]
    assert batch["full_references"] == old["full_references"]
    assert len(batch["cohorts"]) == len(old["cohorts"]) == 2
    for a, b in zip(batch["cohorts"], old["cohorts"]):
        assert all(
            a[k] == b[k]
            for k in (
                "slots",
                "budgets",
                "generated_ids",
                "target_states",
                "draft_states",
            )
        )
    cli = read(RAW / "model-v3/actual-cli.json")
    old_cli = read(
        ROOT / "benchmarks/v100-mtp-draft-residual/v100/profile-v4/actual-cli.json"
    )
    assert cli["generated_ids"] == old_cli["generated_ids"]
    assert cli["native_memory"]["source_validation_policy"].startswith(
        "resident-forward"
    )
    payloads = read(OUT / "reused-full-payload-pins.json")
    assert payloads["identity"] == terminal["payload_identity"]
    assert not payloads["new_full_target_hash_scan"]
    for label in ("model-v3", "profile-v5"):
        assert (
            read(RAW / label / "payload-before.json")
            == read(RAW / label / "payload-after.json")
            == payloads["identity"]
        )
    campaign = read(RAW / "campaign-v4/campaign-report.json")
    assert (
        campaign["completed"]
        and campaign["model_qualified"]
        and not campaign["fresh_external_campaign_run"]
    )
    assert campaign["library_sha256_by_scope"] == LIBRARIES
    screens = [e for e in campaign["experiments"] if e["label"].startswith("screen-")]
    assert len(screens) == 18
    for name in ("mtp2", "mtp-batch", "ordinary256"):
        cells = [
            e for e in screens if e["label"].startswith("screen-" + name + "-trial")
        ]
        assert [e["source_scope_mode"] for e in cells] == [0, 1, 1, 0, 0, 1]
    for e in campaign["experiments"]:
        assert (
            e["completed"]
            and e["stable_outputs"]
            and e["library_sha256"] == LIBRARIES[str(e["source_scope_mode"])]
        )
        assert e["health"]["source_validation_policy"].startswith(
            "resident-forward"
        ) == bool(e["source_scope_mode"])
        b = read(RAW / "campaign-v4" / e["label"] / "timed/report.json")
        assert all(
            w["emitted_tokens"] == 32 * w["request_count"]
            and all(
                r["response"]["tokens"] == b["case"]["expected_generated_ids"]
                for r in w["requests"]
            )
            for w in b["waves"]
        )
        assert len(b["waves"]) == (2 if e["label"].startswith("screen-") else 12)
    counts = read(RAW / "profile-v5/scoped-syscalls.json")
    assert counts["engine_change"] and counts["source_check_cadence_changed"]
    assert (
        not counts["failures"]
        and not counts["unmatched_pending_tids"]
        and counts["model_outputs_exact"]
    )
    assert (
        0
        < counts["checkpoint_calls"]["descriptor"]
        == counts["checkpoint_calls"]["pathname"]
        < 6674
    )
    raw_trace = read(RAW / "profile-v5/scoped-metadata.json")
    assert (
        not raw_trace["source_guards_unchanged"]
        and raw_trace["source_integrity_predicate_preserved"]
    )
    for scope in [0, 1]:
        resource = read(RAW / "profile-v5" / f"scope{scope}-resources.json")
        assert resource["library_sha256"] == LIBRARIES[str(scope)]
        assert (
            resource["achieved_occupancy"] is None
            and not resource["hardware_counter_collection"]
        )
        for mode in ("mtp2-c1", "mtp-batch-c8"):
            meta = read(RAW / "profile-v5" / f"scope{scope}-{mode}-metadata.json")
            assert meta["profiled"] and meta["library_sha256"] == LIBRARIES[str(scope)]
            assert meta["source_validation_policy"].startswith(
                "resident-forward"
            ) == bool(scope)
    metrics = read(OUT / "metrics.json")
    assert (
        not metrics["external_speed_goal_complete"]
        and metrics["full_goal_scope_not_redefined"]
    )
    assert (
        metrics["source_check_audit"]["predicate_preserved"]
        and metrics["source_check_audit"]["cadence_changed"]
    )
    assert not metrics["prototype_promoted_to_production"]
    assert metrics["production_restoration"] == restoration
    assert not any(c["adopted"] for c in metrics["adoption_checks"].values())
    files = []
    for directory in ["src", "tests", "tools", "scripts", "docs"]:
        files.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files.extend(ROOT / p for p in ["README.md", "pyproject.toml", "AGENTS.md"])
    files.extend(
        OUT / p
        for p in [
            "summary.md",
            "design.md",
            "summarize.py",
            "freeze.py",
            "reject_prototype.py",
        ]
    )
    source = {}
    for p in sorted(set(files)):
        rel = p.relative_to(ROOT)
        dst = OUT / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        source[str(rel)] = sha(p)
    (OUT / "source-sha256.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n"
    )
    checks = dict(
        frozen=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_files=len(source),
        prior_capsules_verified=40,
        prior_verification=verified,
        library_sha256_by_scope=LIBRARIES,
        original_default_is_per_weight=True,
        source_guard_predicate_preserved=True,
        changed_resident_detection_cadence_declared=True,
        deterministic_mutation_and_scope_ownership_tests_passed=True,
        preserved_initial_metadata_accessor_test_failure=True,
        native_full_suite_passed=2269,
        local_final_suite_passed=835,
        local_restored_production_suite_passed=822,
        production_source_identical_to_pre_experiment=True,
        prototype_only_not_shipped=True,
        trained_trajectories_private_states_and_corrected_draft_exact=True,
        independent_arithmetic_quality_reference_explicitly_reused=True,
        source_syscall_reduction_measured_not_inferred_speed=True,
        adoption_checks=metrics["adoption_checks"],
        separate_paired_profiles=True,
        achieved_occupancy=None,
        counter_policy_unchanged=True,
        all_five_workers_terminal=True,
        original_gpu_footprint_restored=True,
        full_goal_complete=False,
        no_local_rust_cuda_compilation=True,
        no_commit_or_push=True,
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
    print(
        json.dumps(
            dict(
                frozen=True,
                source_files=len(source),
                evidence_files=len(evidence),
                prior_capsules_verified=40,
                adoption_checks=metrics["adoption_checks"],
                full_goal_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
