"""Freeze the corrected positive operator pilot, not an integrated engine gain."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / "v100"
BLOCKS = "6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2"
HOST = "202a38580846334ac9d08893b4f85064f6f5eae8f65f451c2200cfe0c78fa8e1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "evidence-sha256.json").exists(), "Never replace frozen evidence"
    prior = read(OUT / "prior-capsule-audit.json")
    assert len(prior) == 41
    for rel, h in read(OUT / "prior-reference-pins.json").items():
        assert sha(ROOT / rel) == h, rel
    verified = {}
    for name in prior:
        cap = ROOT / "benchmarks" / name
        count = 0
        missing = []
        for filename, base in [
            ("source-sha256.json", cap / "source"),
            ("evidence-sha256.json", cap),
        ]:
            if not (cap / filename).exists():
                continue
            manifest = read(cap / filename)
            manifest = manifest.get("files", manifest)
            found = 0
            for rel, h in manifest.items():
                p = base / rel
                if not p.is_file() and name == "decoder-layer":
                    p = base / Path(rel).name
                if p.is_file():
                    assert sha(p) == h, str(p)
                    count += 1
                    found += 1
                else:
                    missing.append(rel)
            assert not missing or (name == "decoder-layer" and found == 4)
        verified[name] = dict(verified_files=count, legacy_missing=missing)
    terminal = read(OUT / "terminal-audit.json")
    expected = {
        "fixtures-v1": 1,
        "fixtures-v2": 1,
        "fixtures-v3": 0,
        "trained-v4": 1,
        "trained-v5": 0,
        "profile-v6": 0,
        "fixtures-v7": 0,
        "trained-v8": 0,
        "profile-v9": 0,
    }
    assert {k: v["exit_code"] for k, v in terminal["workers"].items()} == expected
    assert (
        not terminal["owned_processes"]
        and not terminal["remaining_periodic_samplers"]
        and not terminal["port18953"]
    )
    assert "3651 MiB" in terminal["gpu_after"] and "3189 MiB" in terminal["gpu_after"]
    assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
    for n in range(1, 5):
        manifest = read(OUT / f"source-manifest-v{n}.json")
        assert manifest == terminal["source_manifests"][str(n)]
        assert all(
            sha(OUT / f"executed-source-v{n}" / rel) == h for rel, h in manifest.items()
        )
    latest = read(OUT / "source-manifest-v4.json")
    current = {
        str(p.relative_to(ROOT)): sha(p)
        for d in ["src", "tests", "tools"]
        for p in (ROOT / d).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    assert current == latest
    old = read(ROOT / "benchmarks/v100-source-validation/source-manifest-v1.json")
    assert all(current[k] == h for k, h in old.items() if k.startswith("src/"))
    changes = read(OUT / "production-unchanged.json")
    assert changes["production_src_unchanged"] and len(changes["changes"]) == 5
    build = read(RAW / "fixtures-v7/build.json")
    assert (
        build["library_sha256"] == HOST
        and build["blocks_library_sha256"] == BLOCKS
        and build["no_new_cuda_kernels"]
    )
    assert build["host_cpp_sha256"] == sha(ROOT / "tools/projection/gdn_head_pilot.cpp")
    blocks_build = read(
        ROOT
        / "benchmarks/v100-mtp-draft-residual/v100/fixtures-v1/build/089e914d53d145f0/build.json"
    )
    assert (
        blocks_build["library_sha256"] == BLOCKS
        and len(blocks_build["source_sha256"]) == 10
    )
    assert all(
        sha(ROOT / "src/venturi/ops/csrc" / r) == h
        for r, h in blocks_build["source_sha256"].items()
    )
    assert "invalid" in (RAW / "fixtures-v1/host-build.log").read_text()
    assert (
        "Pilot requires aligned" in (RAW / "fixtures-v2/focused-tests.log").read_text()
    )
    assert "FileNotFoundError" in (RAW / "trained-v4/benchmark.log").read_text()
    for tag in ["fixtures-v3", "fixtures-v7"]:
        assert "9 passed" in (RAW / tag / "focused-tests.log").read_text()
        assert (
            "817 passed, 2119 skipped"
            in (RAW / tag / "cpu-and-pilot-regression.log").read_text()
        )
    assert "828 passed, 2108 skipped" in (OUT / "local-tests-v3.log").read_text()
    assert "2 passed" in (OUT / "local-parser-tests.log").read_text()
    trained = read(RAW / "trained-v8/measurement/report.json")
    assert (
        trained["completed"]
        and not trained["model_integration"]
        and not trained["model_tps_claim"]
        and trained["integration_gate"]["passed"]
        and len(trained["cells"]) == 6
    )
    assert trained["host_library_sha256"] == HOST and trained["blocks_sha256"] == BLOCKS
    assert trained["artifact_id"] == "3d635b87db004f42aefd6730e372e530"
    source = read(RAW / "trained-v8/verified-source-files.json")
    assert source["official_revision"] == "fc05daec18b0a78c049392ed2e771dde82bdf654"
    assert source["identity_before"] == read(RAW / "trained-v8/source-after.json")
    original = {f["name"]: f for f in source["source_manifest"]["files"]}
    for name, h in source["fresh_sha256"].items():
        if name in original:
            assert h == original[name]["actual_sha256"], name
        if name.endswith(".safetensors"):
            assert h == original[name]["lfs_sha256"]
    profile = read(RAW / "profile-v9/completed.json")
    assert (
        profile["completed"]
        and profile["profiled_separately"]
        and profile["counters_matched_trace"]
    )
    assert "2 passed" in (RAW / "profile-v9/parser-tests.log").read_text()
    psource = read(RAW / "profile-v9/verified-source-files.json")
    assert psource["fresh_sha256"] == source["fresh_sha256"] and psource[
        "identity_before"
    ] == source["identity_before"] == read(RAW / "profile-v9/source-after.json")
    metrics = read(OUT / "metrics.json")
    assert (
        metrics["corrected_schedule"]
        and metrics["integration_gate"] == trained["integration_gate"]
        and not metrics["new_model_tps"]
    )
    for cell in metrics["cells"]:
        assert (
            len(cell["paired_trial_speedups"]) == 3
            and min(cell["paired_trial_speedups"]) > 1.10
        )
        assert cell["projected_output_relative_l2_vs_row"] < 5e-4
        assert (
            cell["source_reference_max_relative_l2"] <= 0.02
            and cell["source_reference_min_cosine"] >= 0.999
        )
    assert len(metrics["separate_profile"]) == 6
    pinned = read(OUT / "reused-full-payload-pins.json")
    assert pinned["identity"] == terminal["payload_identity"]
    for tag in ["trained-v5", "trained-v8", "profile-v6", "profile-v9"]:
        assert read(RAW / tag / "payload-before.json") == read(
            RAW / tag / "payload-after.json"
        )
    files = [ROOT / rel for rel in latest] + [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "docs/native-q8-goal.md",
        ROOT / "docs/native-mtp-plan.md",
    ]
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in (".md", ".py")]
    sources = {}
    for p in sorted(set(files)):
        rel = p.relative_to(ROOT)
        dst = OUT / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        sources[str(rel)] = sha(p)
    (OUT / "source-sha256.json").write_text(
        json.dumps(sources, indent=2, sort_keys=True) + "\n"
    )
    check = dict(
        frozen=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_files=len(sources),
        prior_capsules_verified=41,
        prior_verification=verified,
        corrected_production_gate_dispatch=True,
        preserved_three_initial_failures=True,
        preserved_initial_schedule_results=True,
        focused_two_gpu_tests_passed=9,
        remote_cpu_and_pilot_tests_passed=817,
        remote_skipped=2119,
        local_tests_passed=828,
        local_skipped=2108,
        new_native_full_engine_regression_campaign=False,
        production_src_unchanged=True,
        qualified_blocks_sha256=BLOCKS,
        research_host_sha256=HOST,
        trained_operator_gate_passed=True,
        whole_model_integration=False,
        new_model_tps=False,
        achieved_hardware_counters=None,
        counter_policy_unchanged=True,
        nine_workers_terminal=True,
        original_gpu_footprint_restored=True,
        full_goal_complete=False,
        no_local_rust_cuda_compilation=True,
        no_commit_or_push=True,
    )
    (OUT / "evidence-check.json").write_text(json.dumps(check, indent=2) + "\n")
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
                source_files=len(sources),
                evidence_files=len(evidence),
                prior_capsules_verified=41,
                positive_operator_gate=True,
                full_goal_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
