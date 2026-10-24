"""Freeze a rejected operator prerequisite without inventing a model speed gain."""

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
    assert len(prior) == 37
    for rel, digest in read(OUT / "prior-reference-pins.json").items():
        assert sha(ROOT / rel) == digest, rel
    prior_counts = {}
    for name in prior:
        cap = ROOT / "benchmarks" / name
        count, missing = 0, []
        for filename, base in [
            ("source-sha256.json", cap / "source"),
            ("evidence-sha256.json", cap),
        ]:
            if not (cap / filename).is_file():
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
        prior_counts[name] = dict(verified_files=count, legacy_missing=missing)
    terminal = read(OUT / "terminal-audit.json")
    assert all(w["exit_code"] == 0 for w in terminal["workers"].values())
    assert (
        not terminal["owned_processes"] and not terminal["remaining_periodic_samplers"]
    )
    assert "3651 MiB" in terminal["gpu_after"] and "3189 MiB" in terminal["gpu_after"]
    assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
    for version in [1, 2]:
        manifest = read(OUT / f"source-manifest-v{version}.json")
        remote = (
            f"/home/data/wangyue/repos/codex-build/Venturi/dp2a-pilot-src-v{version}"
        )
        assert manifest == terminal["snapshots"][remote]
        assert all(
            sha(OUT / f"executed-source-v{version}" / p) == h
            for p, h in manifest.items()
        )
    assert all(
        sha(ROOT / p) == h for p, h in read(OUT / "source-manifest-v2.json").items()
    )
    raw = OUT / "v100"
    fixture = raw / "fixtures-v1"
    campaign = raw / "campaign-v2"
    build = read(fixture / "build.json")
    assert (
        build["library_sha256"]
        == "96b41c84b209a97e0b7aa196e1b2cedb2e6d883ce55e4b0c79532eebffd2c876"
    )
    assert all(sha(ROOT / p) == h for p, h in build["source_sha256"].items())
    assert "59 passed" in (fixture / "tests.log").read_text()
    assert "12 passed" in (campaign / "occupancy-tool-tests.log").read_text()
    assert "815 passed, 2076 skipped" in (OUT / "local-tests.log").read_text()
    assert "All checks passed!" in (OUT / "ruff.log").read_text()
    reused = read(campaign / "reused-fixtures.json")
    assert reused["test_log_sha256"] == sha(fixture / "tests.log")
    assert reused["library_sha256"] == build["library_sha256"]
    # Research additions must not silently alter the qualified production model.
    qualified = ROOT / "benchmarks/v100-native-column-mlp"
    previous = read(qualified / "source-sha256.json")
    assert all(sha(ROOT / p) == h for p, h in previous.items() if p.startswith("src/"))
    old_build = read(next((qualified / "v100/fixtures-v1/build").rglob("build.json")))
    assert old_build["library_sha256"] == build["control_library_sha256"]
    assert len(old_build["source_sha256"]) == 10
    assert all(
        sha(ROOT / "src/venturi/ops/csrc" / p) == h
        for p, h in old_build["source_sha256"].items()
    )
    assert "q8_dp2a.cuh" not in (ROOT / "src/venturi/ops/csrc/blocks.cu").read_text()
    timed = read(campaign / "timed/report.json")
    assert timed["completed"] and not timed["profiled"] and len(timed["cases"]) == 18
    assert not timed["promote_to_model"] and not timed["model_dispatch_changed"]
    for case in timed["cases"]:
        assert len(case["waves"]) == 6
        for arm in [0, 1]:
            assert len([w for w in case["waves"] if w["arm"] == arm]) == 3
        assert all(w["calls"] == 1024 for w in case["waves"])
        assert all(
            c["relative_l2"]
            < (2e-6 if case["output_dtype"] == "torch.float32" else 0.001)
            and c["versus_unquantized_fp16_activation_fp64"]["relative_l2"] < 0.002
            for c in case["reference_checks"]
        )
    profile = read(campaign / "profile/report.json")
    assert profile["completed"] and profile["profiled"] and len(profile["cases"]) == 6
    attribution = read(campaign / "attribution.json")
    assert len(attribution["ranges"]) == 12
    assert all(r["launches"] == 32 for r in attribution["ranges"])
    for name, digest in [
        ("pilot", build["library_sha256"]),
        ("control", old_build["library_sha256"]),
    ]:
        r = read(campaign / (name + "-resources.json"))
        assert r["library_sha256"] == digest and r["achieved_occupancy"] is None
        assert (
            not r["hardware_counter_collection"]
            and r["kernel_launches_by_this_tool"] == 0
        )
        for c in r["devices"][0]["traces"]["projection"]:
            for launch in c["launches"]:
                expected = (
                    "4vi16" in launch["symbol"]
                    if name == "pilot"
                    else "4vi16" not in launch["symbol"]
                )
                assert (
                    launch["status"]
                    == "Exact symbol and trace register/shared usage match"
                ) == expected
    sass = read(campaign / "pilot-sass-integer-evidence.json")
    assert sass["idp2a_instructions"] > 0 and sass["idp4a_instructions"] == 0
    metrics = read(OUT / "metrics.json")
    assert (
        metrics["median_fused_operator_ratio"] < 1.05
        and not metrics["promote_to_model"]
    )
    assert not metrics["full_goal_complete"] and not metrics["new_model_tps_measured"]
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
        created_utc=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn="progress: native column-MLP model gain and qualified repeat matrices",
        source_files=len(manifest),
        prior_capsules_verified=37,
        prior_verification=prior_counts,
        independent_operator_correctness_passed=True,
        remote_fixture_tests=59,
        remote_resource_parser_tests=12,
        local_passed=815,
        local_skipped=2076,
        trained_checkpoint_operator_cells=18,
        repeats_per_arm_cell=3,
        pilot_library_sha256=build["library_sha256"],
        production_library_sha256=old_build["library_sha256"],
        qualified_production_source_unchanged=True,
        candidate_promoted_to_model=False,
        new_model_throughput_claim=False,
        full_goal_complete=False,
        achieved_occupancy=None,
        counter_policy_unchanged=True,
        all_owned_workers_terminal=True,
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
    assert all(sha(OUT / "source" / p) == h for p, h in manifest.items())
    print(
        json.dumps(
            dict(
                frozen=True,
                source_files=len(manifest),
                evidence_files=len(evidence),
                prior_capsules_verified=37,
                candidate_promoted=False,
                goal_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
