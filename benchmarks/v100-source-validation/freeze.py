"""Freeze an unchanged-engine diagnostic without claiming an optimization win."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / "v100/probe-v1"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT / "evidence-sha256.json").exists(), "Never replace frozen evidence"
    prior = read(OUT / "prior-capsule-audit.json")
    assert len(prior) == 39
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
                path = base / rel
                if not path.is_file() and name == "decoder-layer":
                    path = base / Path(rel).name
                if path.is_file():
                    assert sha(path) == digest, str(path)
                    count += 1
                    found += 1
                else:
                    missing.append(rel)
            assert not missing or (name == "decoder-layer" and found == 4)
        verified[name] = dict(verified_files=count, legacy_missing=missing)
    old = ROOT / "benchmarks/v100-mtp-draft-residual"
    for rel, digest in read(old / "source-manifest-v1.json").items():
        if rel.startswith("src/"):
            assert sha(ROOT / rel) == digest, "Inference core changed: " + rel
    terminal = read(OUT / "terminal-audit.json")
    assert terminal["exit_code"] == 0
    assert (
        not terminal["owned_processes"]
        and not terminal["owned_sampler_live"]
        and not terminal["port18953"]
    )
    assert "3651 MiB" in terminal["gpu_after"] and "3189 MiB" in terminal["gpu_after"]
    assert "RmProfilingAdminOnly: 1" in terminal["counter_policy"]
    manifest = read(OUT / "source-manifest-v1.json")
    assert len(manifest) == 289 and terminal["source_manifest"] == manifest
    assert manifest == read(RAW / "executed-source.json")
    assert all(sha(ROOT / rel) == digest for rel, digest in manifest.items())
    assert all(
        sha(OUT / "executed-source-v1" / rel) == digest
        for rel, digest in manifest.items()
    )
    digest = "6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2"
    completed = read(RAW / "completed.json")
    assert (
        completed["completed"]
        and completed["engine_unchanged"]
        and completed["no_source_guards_removed"]
    )
    assert completed["library_sha256"] == digest == terminal["library_sha256"]
    assert (RAW / "exit-code.txt").read_text().strip() == "0"
    pins = read(old / "reused-full-payload-pins.json")
    before, after = read(RAW / "payload-before.json"), read(RAW / "payload-after.json")
    assert before == after == pins["identity"]
    assert "822 passed, 2105 skipped" in (OUT / "local-tests.log").read_text()
    assert "4 passed" in (RAW / "parser-tests.log").read_text()
    assert "All checks passed!" in (OUT / "ruff.log").read_text()
    assert read(RAW / "cpu-build.log.command.json")[0] == "g++"
    assert len((RAW / "cpu-binary-sha256.txt").read_text().strip()) == 64
    trace_command = read(RAW / "traced.log.command.json")
    assert trace_command[0] == "/usr/bin/strace" and "-p" not in trace_command
    assert "trace=fstat,stat,newfstatat,statx" in trace_command
    expected = read(old / "v100/profile-v4/actual-cli.json")["generated_ids"]
    for name, count, instrumented in [("untraced", 3, False), ("traced", 1, True)]:
        r = read(RAW / (name + ".json"))
        assert r["library_sha256"] == digest and r["source_guards_unchanged"]
        assert r["instrumented"] == instrumented and len(r["calls"]) == count
        assert r["prepared_slots"] == 8 and r["capacity_per_slot"] == 256
        assert r["projection_rows"] == 32 and not r["python_math_in_marked_call"]
        for call in r["calls"]:
            assert call["output_ids_equal"] and call["output_ids"] == expected
            s = call["stats"]
            assert (s["accepted_tokens"], s["proposed_tokens"]) == (17, 26)
            assert (
                s["verification_calls"],
                s["target_forward_calls"],
                s["draft_forward_calls"],
            ) == (13, 15, 45)
            assert s["forced_proposals"] == 0 and s["replayed_input_tokens"] == 0
    syscalls = read(RAW / "syscall-report.json")
    assert syscalls["checkpoint_calls"] == dict(descriptor=6674, pathname=6674)
    assert syscalls["all_traced_metadata_calls_in_intervals"] == 13348
    assert not syscalls["failures"] and not syscalls["unmatched_pending_tids"]
    assert not syscalls["engine_change"] and syscalls["model_outputs_exact"]
    for name in ("target", "draft"):
        micro = read(RAW / (name + "-microbench.json"))
        assert (
            micro["verification_pairs_per_repeat"] == 100000
            and len(micro["seconds"]) == 4
        )
        assert all(0 < value < 10 for value in micro["seconds"])
    metrics = read(OUT / "metrics.json")
    assert metrics["eligibility_gate"]["eligible_for_followup_test"]
    assert not metrics["eligibility_gate"]["implementation_or_adoption_gate_passed"]
    assert metrics["opportunity_is_not_critical_path_or_tps_prediction"]
    assert (
        not metrics["external_speed_goal_complete"]
        and not metrics["inference_optimization_implemented_in_this_capsule"]
    )
    for rel, h in metrics[
        "unchanged_library_prior_timeline_and_goal_references"
    ].items():
        assert sha(ROOT / rel) == h
    for mode in ("untraced_telemetry", "instrumented_telemetry"):
        assert all(metrics[mode][str(i)]["samples"] > 0 for i in (0, 1))
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
        prior_capsules_verified=39,
        prior_verification=verified,
        unchanged_inference_library_sha256=digest,
        no_source_guards_removed=True,
        all_marked_outputs_and_counters_exact=True,
        metadata_syscalls_measured=13348,
        independent_cpu_microbench_is_not_inference_critical_path=True,
        eligibility_gate_only_passed=True,
        optimization_implemented=False,
        profile_counters_and_achieved_occupancy_unavailable=True,
        counter_policy_unchanged=True,
        owned_processes_and_sampler_terminal=True,
        original_gpu_footprint_restored=True,
        full_goal_complete=False,
        no_local_rust_cuda_or_cpp_compilation=True,
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
                prior_capsules_verified=39,
                engine_change=False,
                full_goal_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
