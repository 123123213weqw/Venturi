"""Freeze qualified native integration and measured gaps without declaring victory."""

import datetime
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PROFILE = "q8-residual-column-mlp"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT / "evidence-sha256.json").exists(), (
        "Never overwrite frozen evidence"
    )
    prior = read(OUT / "prior-capsule-audit.json")
    assert len(prior) == 36
    for rel, h in read(OUT / "prior-reference-pins.json").items():
        assert sha(ROOT / rel) == h, rel
    for name in prior:
        cap = ROOT / "benchmarks" / name
        for manifest, base in [
            ("source-sha256.json", cap / "source"),
            ("evidence-sha256.json", cap),
        ]:
            if not (cap / manifest).exists():
                continue
            m = read(cap / manifest)
            m = m.get("files", m)
            found, missing = 0, []
            for rel, h in m.items():
                p = base / rel
                if not p.exists() and name == "decoder-layer":
                    p = base / Path(rel).name
                if p.is_file():
                    assert sha(p) == h, str(p)
                    found += 1
                else:
                    missing.append(rel)
            assert not missing or (name == "decoder-layer" and found == 4)
    terminal = read(OUT / "terminal-audit.json")
    expected = {
        "fixtures-v1": 0,
        "model-v2": 0,
        "campaign-v3": 143,
        "budget-v4": 1,
        "budget-v5": 0,
        "campaign-v6": 0,
        "profile-v7": 1,
        "cli-v8": 1,
        "profile-v9": 0,
        "cli-v10": 0,
    }
    assert {k: v["exit_code"] for k, v in terminal["workers"].items()} == expected
    assert (
        not terminal["owned_inference_processes"]
        and not terminal["remaining_periodic_nvml_samplers"]
        and not terminal["listening_ports"]
    )
    assert "3651 MiB" in terminal["gpu_after"] and "3189 MiB" in terminal["gpu_after"]
    assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
    for v in range(1, 6):
        snap = f"native-column-mlp-src-v{v}"
        m = read(OUT / f"source-manifest-v{v}.json")
        assert m == terminal["executed_snapshots"][snap]
        assert all(
            sha(OUT / "executed-source-deltas" / snap / rel) == h
            for rel, h in m.items()
        )
    current = read(OUT / "source-manifest-v5.json")
    assert all(
        sha(ROOT / rel) == h
        for rel, h in current.items()
        if rel.startswith(("src/", "tests/", "tools/", "scripts/"))
        or rel == "pyproject.toml"
    ), "Code changed after executed snapshot5"
    pins = read(OUT / "fresh-payload-and-external-bundle-pins.json")
    assert pins["fresh_after_all_inference_jobs"]
    assert pins["all_eighty_release_files_match"]
    assert len(pins["pinned_release_files"]) == 80
    assert all(
        p["matches"] and p["expected_sha256"] == p["actual_sha256"]
        for p in pins["pinned_release_files"].values()
    )
    assert pins["prior_release_pin_report_sha256"] == sha(
        ROOT / "benchmarks/v100-llama-comparison/v100/screen-report.json"
    )
    assert {p["sha256"] for p in pins["checkpoint_payloads"].values()} == {
        "b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8",
        "a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641",
        "eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513",
    }
    compatibility = read(OUT / "external-control-compatibility.json")
    assert compatibility["matched"] and len(compatibility["checks"]) == 2
    assert compatibility["fresh_payload_and_bundle_pins_sha256"] == sha(
        OUT / "fresh-payload-and-external-bundle-pins.json"
    )
    raw = OUT / "v100"
    assert "12 passed" in (raw / "fixtures-v1/focused-tests.log").read_text()
    assert (
        "2132 passed, 687 skipped" in (raw / "fixtures-v1/full-tests.log").read_text()
    )
    assert "22 passed" in (raw / "model-v2/expanded-tests.log").read_text()
    assert "803 passed, 2028 skipped" in (OUT / "local-tests-final.log").read_text()
    assert "All checks passed!" in (OUT / "ruff.log").read_text()
    b = read(next((raw / "fixtures-v1/build").rglob("build.json")))
    h = b["library_sha256"]
    assert len(b["source_sha256"]) == 10
    assert all(
        sha(ROOT / "src/venturi/ops/csrc" / rel) == v
        for rel, v in b["source_sha256"].items()
    )
    q = read(raw / "model-v2/qualification/report.json")
    assert q["completed"] and q["library_sha256"] == h
    for name in ["fp16", PROFILE]:
        c = q["profiles"][name]["candidate"]
        assert c["passed"] and len(c["cases"]) == 4
        assert sum(len(v["all_logit_bytes"]) for v in c["cases"]) == 98
        assert (
            len(c["batch"]["private_cache"])
            == len(c["reset_continuation"]["private_cache"])
            == 8
        )
    assert all(
        v["passed"]
        for c in q["profiles"][PROFILE]["candidate"]["cases"]
        for v in c["teacher_logit_checks"]
    )
    teacher_path = (
        ROOT
        / "benchmarks/v100-q8-shared-activation/v100/model-v2/qualification/report.json"
    )
    teacher = read(teacher_path)
    assert q["prior_teacher_report_sha256"] == sha(teacher_path)
    a, b = (p["profiles"]["fp16"]["candidate"] for p in [q, teacher])
    for x, y in zip(a["cases"], b["cases"]):
        for key in ["generated_ids", "all_logit_bytes", "committed_private_state"]:
            assert x[key] == y[key]
    assert a["batch"] == b["batch"]
    assert a["reset_continuation"] == b["reset_continuation"]
    q = read(raw / "budget-v5/qualification/report.json")
    assert (
        q["completed"]
        and q["same_profile_row_budget_bytes_equal"]
        and q["fixed_work"]["max_tokens"] == 256
    )
    q = read(raw / "model-v2/mtp-batch-qualification/report.json")
    assert (
        q["completed"]
        and q["selective_reset_untouched_state_exact"]
        and len(q["cohorts"]) == 2
    )
    assert all(
        c["exact_target_state"]
        and c["exact_corrected_draft_state"]
        and c["native_without_python_math"]
        for c in q["cohorts"]
    )
    q = read(raw / "model-v2/mtp2-qualification/report.json")
    assert q["completed"] and len(q["cases"]) == 4
    assert all(
        c["output_ids_equal"]
        and c["committed_target_state_bytes_equal"]
        and c["committed_draft_state_equal_mtp1"]
        for c in q["cases"]
    )
    cli = read(raw / "cli-v10/actual-cli.json")
    assert (
        cli["projection_profile"] == PROFILE
        and cli["native_memory"]["column_mlp_enabled"]
    )
    assert read(raw / "cli-v10/completed.json")["actual_public_cli_fixed_tokens_match"]
    assert "11 passed" in (raw / "cli-v10/cli-tests.log").read_text()
    campaign = read(raw / "campaign-v6/campaign-report.json")
    assert campaign["completed"] and not campaign["external_goal_complete"]
    for e in campaign["experiments"]:
        assert e["completed"] and e["stable_outputs"]
        r = read(raw / "campaign-v6" / e["label"] / "timed/report.json")
        assert all(
            w["emitted_tokens"] == 32 * w["request_count"]
            and all(
                x["response"]["tokens"] == r["case"]["expected_generated_ids"]
                for x in w["requests"]
            )
            for w in r["waves"]
        )
    resources = read(raw / "profile-v9/resources.json")
    assert (
        resources["library_sha256"] == h
        and resources["achieved_occupancy"] is None
        and not resources["hardware_counter_collection"]
    )
    for name in [
        "same-library-row-mtp2-c1",
        "column-mtp2-c1",
        "column-ordinary256-c8",
        "column-mtp-batch-c8",
    ]:
        assert read(raw / "profile-v9" / (name + "-metadata.json"))["profiled"]
        assert not read(raw / "profile-v9" / (name + "-attribution.json"))[
            "achieved_hardware_counters"
        ]
    metrics = read(OUT / "metrics.json")
    assert not metrics["external_speed_goal_complete"]
    decision = read(OUT / "decision.json")
    assert decision["metrics_sha256"] == sha(OUT / "metrics.json")
    assert decision["adopt_as_explicit_opt_in"]
    assert not decision["change_existing_defaults"]
    assert decision["complete_http_matrix_goal_met"]
    assert not decision["full_external_speed_goal_complete"]
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
            "next-bottlenecks.md",
            "summarize.py",
            "check_external_compatibility.py",
            "make_decision.py",
            "freeze.py",
        ]
    )
    m = {}
    for p in sorted(set(files)):
        rel = p.relative_to(ROOT)
        dst = OUT / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        m[str(rel)] = sha(p)
    (OUT / "source-sha256.json").write_text(
        json.dumps(m, indent=2, sort_keys=True) + "\n"
    )
    checks = dict(
        frozen=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_files=len(m),
        prior_capsules_verified=36,
        native_library_sha256=h,
        correctness_qualified=True,
        private_cache_and_corrected_draft_gates_passed=True,
        projection_row_budgets_32_256_exact=True,
        actual_public_cli_verified=True,
        full_payloads_and_all_eighty_external_release_files_rehashed=True,
        stronger_external_control_compatibility_checked=True,
        complete_http_matrix_goal_met=True,
        compute_profile=PROFILE,
        changed_fp32_association_declared=True,
        original_defaults_unchanged=True,
        full_external_speed_goal_complete=False,
        achieved_occupancy=None,
        all_owned_processes_terminal=True,
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
    assert all(sha(OUT / p) == v for p, v in evidence.items())
    assert all(sha(OUT / "source" / p) == v for p, v in m.items())
    print(
        json.dumps(
            dict(
                frozen=True,
                source_files=len(m),
                evidence_files=len(evidence),
                prior_capsules_verified=36,
                goal_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
