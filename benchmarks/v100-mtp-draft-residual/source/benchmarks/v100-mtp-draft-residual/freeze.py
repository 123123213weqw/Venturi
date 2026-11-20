"""Freeze the qualified explicit draft option without shrinking the full goal."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "evidence-sha256.json").exists(), "Never replace frozen evidence"
    prior = read(OUT / "prior-capsule-audit.json")
    assert len(prior) == 38
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
        "fixtures-v1": 0,
        "model-v2": 0,
        "campaign-v3": 0,
        "profile-v4": 0,
        "guards-v5": 1,
        "guards-v6": 0,
    }
    assert (
        not terminal["owned_processes"]
        and not terminal["remaining_periodic_samplers"]
        and not terminal["port18953"]
    )
    assert "3651 MiB" in terminal["gpu_after"] and "3189 MiB" in terminal["gpu_after"]
    assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
    manifest = read(OUT / "source-manifest-v1.json")
    assert terminal["source_manifest"] == manifest
    assert all(sha(ROOT / rel) == digest for rel, digest in manifest.items())
    assert all(
        sha(OUT / "executed-source-v1" / rel) == digest
        for rel, digest in manifest.items()
    )
    raw = OUT / "v100"
    assert "32 passed" in (raw / "fixtures-v1/focused-tests.log").read_text()
    assert (
        "2236 passed, 687 skipped" in (raw / "fixtures-v1/full-tests.log").read_text()
    )
    assert "818 passed, 2105 skipped" in (OUT / "local-tests.log").read_text()
    assert "All checks passed!" in (OUT / "ruff.log").read_text()
    build = read(next((raw / "fixtures-v1/build").rglob("build.json")))
    digest = build["library_sha256"]
    assert digest == "6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2"
    assert len(build["source_sha256"]) == 10
    assert all(
        sha(ROOT / "src/venturi/ops/csrc" / p) == h
        for p, h in build["source_sha256"].items()
    )
    target_pin = (
        ROOT
        / "benchmarks/v100-native-column-mlp/v100/model-v2/qualification/report.json"
    )
    target = read(target_pin)
    assert target["completed"]
    assert (
        sum(
            len(c["all_logit_bytes"])
            for c in target["profiles"]["q8-residual-column-mlp"]["candidate"]["cases"]
        )
        == 98
    )
    for profile in ["fp16", "q8-residual"]:
        q = read(raw / "model-v2" / (profile + "-qualification/report.json"))
        assert q["completed"] and q["library_sha256"] == digest
        assert q["mtp_draft_projection_profile"] == profile
        assert q["prior_target_report_sha256"] == sha(target_pin)
        assert (
            q["mtp_artifact_sha256"]
            == "a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641"
        )
        assert len(q["block_reference"]) == 2 and len(q["recurrence_reference"]) == 3
        assert all(
            c["passed"] for c in q["block_reference"] + q["recurrence_reference"]
        )
        assert len(q["cases"]) == 4
        assert all(
            c["output_ids_equal"]
            and c["committed_target_state_bytes_equal"]
            and c["committed_draft_state_equal_mtp1"]
            and c["pinned_target_trajectory_state_reused"]
            and c["native_without_python_math"]
            for c in q["cases"]
        )
        assert [
            (c["mtp_stats"]["accepted_tokens"], c["mtp_stats"]["proposed_tokens"])
            for c in q["cases"]
        ] == [(17, 26), (1, 2), (20, 21), (19, 22)]
    q = read(raw / "model-v2/batch-qualification/report.json")
    assert q["completed"] and q["selective_reset_untouched_state_exact"]
    assert q["profiles"] == dict(
        target="q8-residual-column-mlp", draft="q8-residual", kv="fp16"
    )
    assert len(q["cohorts"]) == 2
    assert all(
        c["exact_target_state"]
        and c["exact_corrected_draft_state"]
        and c["native_without_python_math"]
        for c in q["cohorts"]
    )
    guards = read(raw / "guards-v6/report.json")
    assert guards["completed"] and len(guards["checks"]) == 5
    assert all(c["rejected"] for c in guards["checks"])
    assert sum(c.get("native_guard_without_sdk", False) for c in guards["checks"]) == 1
    assert "FileNotFoundError" in (raw / "guards-v5/checks.log").read_text()
    pins = read(OUT / "reused-full-payload-pins.json")
    assert (
        pins["stable_target_and_draft_identity"]
        and not pins["new_full_target_hash_scan"]
    )
    assert set(pins["full_payload_hashes"].values()) == {
        "b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8",
        "a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641",
    }
    cli = read(raw / "profile-v4/actual-cli.json")
    assert cli["mtp_draft_projection_profile"] == "q8-residual"
    assert cli["native_memory"]["mtp_draft_activation_quantization_passes"] == 2
    assert (
        cli["native_memory"]["mtp_draft_primary_projection_workspace_bytes"] == 1253376
    )
    assert read(raw / "profile-v4/completed.json")[
        "actual_cli_fixed_tokens_and_draft_profile_passed"
    ]
    campaign = read(raw / "campaign-v3/campaign-report.json")
    assert (
        campaign["completed"]
        and campaign["model_qualified"]
        and not campaign["fresh_external_campaign_run"]
    )
    assert all(campaign["path_promotions"].values())
    for e in campaign["experiments"]:
        assert e["completed"] and e["stable_outputs"] and e["library_sha256"] == digest
        assert (
            e["health"]["mtp_draft_projection_profile"]
            == e["mtp_draft_projection_profile"]
        )
        r = read(raw / "campaign-v3" / e["label"] / "timed/report.json")
        assert all(
            w["emitted_tokens"] == 32 * w["request_count"]
            and all(
                x["response"]["tokens"] == r["case"]["expected_generated_ids"]
                for x in w["requests"]
            )
            for w in r["waves"]
        )
    for name in [
        "fp16-draft-mtp2-c1",
        "residual-draft-mtp2-c1",
        "residual-draft-mtp-batch-c8",
    ]:
        metadata = read(raw / "profile-v4" / (name + "-metadata.json"))
        assert metadata["profiled"] and metadata["library_sha256"] == digest
        assert not read(raw / "profile-v4" / (name + "-attribution.json"))[
            "achieved_hardware_counters"
        ]
    resources = read(raw / "profile-v4/resources.json")
    assert (
        resources["library_sha256"] == digest
        and resources["achieved_occupancy"] is None
        and not resources["hardware_counter_collection"]
    )
    metrics = read(OUT / "metrics.json")
    assert all(c["qualifies"] for c in metrics["adoption_checks"].values())
    assert not metrics["external_speed_goal_complete"]
    assert (
        metrics["single_decode_comparison"]["native_tps"]
        < metrics["single_decode_comparison"]["best_known_goal_floor"]
    )
    assert (
        metrics["best_known_native_http_by_count"]["8"]["source"]
        == "prior qualified column-MLP campaign"
    )
    files = []
    for directory in ["src", "tests", "tools", "scripts", "docs"]:
        files.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files.extend(ROOT / p for p in ["README.md", "pyproject.toml", "AGENTS.md"])
    files.extend(
        OUT / p for p in ["summary.md", "design.md", "summarize.py", "freeze.py"]
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
        prior_capsules_verified=38,
        prior_verification=verified,
        library_sha256=digest,
        explicit_residual_draft_qualified=True,
        original_defaults_unchanged=True,
        new_draft_arithmetic_declared=True,
        independent_trained_draft_reference_passed=True,
        target_quality_reference_reused=True,
        same_profile_corrected_state_and_native_source_guards_passed=True,
        three_repeat_changed_paths_passed=True,
        ordinary_c8_and_stronger_external_controls_explicitly_reused=True,
        full_goal_complete=False,
        achieved_occupancy=None,
        counter_policy_unchanged=True,
        all_six_workers_terminal=True,
        preserved_remote_setup_failure=True,
        original_gpu_footprint_restored=True,
        no_local_rust_or_cuda_compilation=True,
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
    assert all(sha(OUT / p) == h for p, h in evidence.items())
    assert all(sha(OUT / "source" / p) == h for p, h in source.items())
    print(
        json.dumps(
            dict(
                frozen=True,
                source_files=len(source),
                evidence_files=len(evidence),
                prior_capsules_verified=38,
                goal_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
