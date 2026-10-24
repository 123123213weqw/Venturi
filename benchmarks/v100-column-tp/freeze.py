"""Freeze the isolated FP32 column/row MLP result without promoting the engine."""

import datetime
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT / "evidence-sha256.json").exists(), (
        "Never overwrite frozen evidence"
    )
    prior = read(OUT / "prior-capsule-audit.json")
    assert len(prior) == 35
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
    a = read(OUT / "terminal-audit.json")
    assert [
        a["workers"][k]["exit_code"]
        for k in ["fixtures-v1", "fixtures-v2", "mlp-v3", "mlp-v4"]
    ] == [0, 1, 0, 0]
    assert (
        not a["owned_inference_processes"]
        and not a["remaining_periodic_nvml_samplers"]
        and not a["listening_ports"]
    )
    assert "3651 MiB" in a["gpu_after"] and "3189 MiB" in a["gpu_after"]
    assert "RmProfilingAdminOnly: 1" in a["counter_policy"]
    for v in [1, 2, 3]:
        snap = f"column-tp-src-v{v}"
        m = read(OUT / f"source-manifest-v{v}.json")
        assert a["executed_snapshots"][snap] == m
        assert all(
            sha(OUT / "executed-source-deltas" / snap / rel) == h
            for rel, h in m.items()
        )
    raw = OUT / "v100"
    assert "32 passed" in (raw / "fixtures-v1/focused-tests.log").read_text()
    assert (
        "2118 passed, 687 skipped" in (raw / "fixtures-v1/full-tests.log").read_text()
    )
    assert "800 passed, 2007 skipped" in (OUT / "local-tests-final.log").read_text()
    assert "All checks passed!" in (OUT / "ruff.log").read_text()
    h = "d7001c142b942af3fe516bc5f00eb828ad2ce24e2716186c9b562301081fb9df"
    b = read(next((raw / "fixtures-v1/build").rglob("build.json")))
    assert b["library_sha256"] == h and len(b["source_sha256"]) == 10
    for rel, value in b["source_sha256"].items():
        assert sha(ROOT / "src/venturi/ops/csrc" / rel) == value
    report = read(raw / "mlp-v4/unprofiled/report.json")
    assert (
        report["completed"] and not report["profiled"] and report["library_sha256"] == h
    )
    assert report["artifact_id"] == "3d635b87db004f42aefd6730e372e530"
    assert (
        not report["model_quality_qualified"]
        and not report["full_model_tp_implemented"]
    )
    for c in report["cases"]:
        assert (
            c["matched_primary_only_input_output"]
            and c["local_gate_up_swiglu_bytes_equal"]
        )
        assert len(c["waves"]) == 6 and c["speed_ratio"] > 1.2
        assert c["difference_l2"] < 2e-6 and c["different_fp16_outputs"] > 0
        assert all(w["calls"] == 400 for w in c["waves"])
        assert all(v["relative_l2"] < 0.001 for v in c["fp64_checks"])
    attr = read(raw / "mlp-v4/attribution.json")
    assert len(attr["operators"]) == 6 and not attr["achieved_hardware_counters"]
    for c in attr["operators"]:
        _, width, mode = c["name"].split("/")
        expected = int(width[1:]) * (84992 if mode == "row" else 30720) * 12
        assert c["transfers"]["host_to_device"]["bytes"] == expected
        assert c["transfers"]["device_to_host"]["bytes"] == expected
        assert c["graph_nodes_actually_attributed"] == c["kernels"]
    resources = read(raw / "mlp-v4/resources.json")
    assert (
        resources["achieved_occupancy"] is None
        and not resources["hardware_counter_collection"]
    )
    for device in resources["devices"]:
        partials = [
            r for r in device["compiled_q8_functions"] if "3vct7partial" in r["symbol"]
        ]
        assert len(partials) == 3 and all(
            r["local_bytes_per_thread"] == 0 for r in partials
        )
    metrics = read(OUT / "metrics.json")
    assert (
        not metrics["external_speed_goal_complete"]
        and metrics["prior_profile_ended_before_matched_unprofiled_trials"]
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
        OUT / p
        for p in [
            "summary.md",
            "design.md",
            "native-integration-contract.md",
            "summarize.py",
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
        prior_capsules_verified=35,
        native_library_sha256=h,
        isolated_operator_qualified=True,
        matched_block_trials=3,
        full_model_quality_qualified=False,
        model_dispatch_changed=False,
        partial_and_transport_precision="FP32",
        changed_fp32_association_declared=True,
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
    assert all(sha(OUT / p) == h for p, h in evidence.items())
    assert all(sha(OUT / "source" / p) == h for p, h in m.items())
    print(
        json.dumps(
            dict(
                frozen=True,
                source_files=len(m),
                evidence_files=len(evidence),
                prior_capsules_verified=35,
                goal_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
