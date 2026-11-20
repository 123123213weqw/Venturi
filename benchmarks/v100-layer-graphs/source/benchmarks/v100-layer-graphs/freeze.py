"""Verify and freeze a negative, correctness-qualified graph experiment."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/"evidence-sha256.json").exists(), "Never overwrite a frozen capsule"
a = read(OUT/"terminal-audit.json")
assert len(a["workers"]) == 6
assert [a["workers"][k]["exit_code"] for k in ["fixtures-v1","fixtures-v2","fixtures-v3","model-v4","campaign-v5","final-v6"]] == [1,1,0,0,0,0]
assert not a["owned_inference_processes"] and not a["remaining_periodic_nvml_samplers"] and not a["listening_ports"]
assert "3651 MiB" in a["gpu_after"] and "3189 MiB" in a["gpu_after"]
assert "RmProfilingAdminOnly: 1" in a["counter_policy"]
for rel,h in read(OUT/"prior-reference-pins.json").items():
    assert sha(ROOT/rel) == h, rel
prior = read(OUT/"prior-capsule-audit.json")
assert len(prior) == 34
for name in prior:
    cap = ROOT/"benchmarks"/name
    for manifest,base in [("source-sha256.json",cap/"source"),("evidence-sha256.json",cap)]:
        if not (cap/manifest).exists():
            continue
        m = read(cap/manifest);m = m.get("files",m);found=0;missing=[]
        for rel,h in m.items():
            p=base/rel
            if not p.exists() and name == "decoder-layer":p=base/Path(rel).name
            if p.is_file():assert sha(p)==h,str(p);found+=1
            else:missing.append(rel)
        assert not missing or (name == "decoder-layer" and found == 4)
for v in [1,2,3]:
    snap=f"layer-graphs-src-v{v}";m=read(OUT/f"source-manifest-v{v}.json")
    assert m == a["executed_snapshots"][snap]
    assert all(sha(OUT/"executed-source-deltas"/snap/rel)==h for rel,h in m.items())
assert "14 passed" in (OUT/"v100/fixtures-v3/mtp-tests.log").read_text()
assert "2105 passed, 674 skipped" in (OUT/"v100/fixtures-v3/full-tests.log").read_text()
assert "792 passed, 1986 skipped" in (OUT/"restored-local-tests.log").read_text()
h="4c997ecd1e45728b83e8e23f1573c9b79a398a3afb6b9e4acfcd97a58cb3eda5"
b=read(next((OUT/"v100/fixtures-v2/build").rglob("build.json")))
assert b["library_sha256"] == h and len(b["source_sha256"]) == 10
assert all(a["executed_snapshots"]["layer-graphs-src-v3"]["src/venturi/ops/csrc/"+rel]==v for rel,v in b["source_sha256"].items())
q=read(OUT/"v100/model-v4/qualification/report.json")
assert q["completed"] and q["candidate_library_sha256"]==h and q["profiles"]["q8-residual"]["same_profile_bytes_equal"]
q=read(OUT/"v100/model-v4/graph-qualification/report.json")
assert q["completed"] and q["library_sha256"]==h and len(q["cases"])==4
assert all(c["all_verification_fp16_bytes_equal"] and c["committed_target_state_equal"] and c["committed_draft_state_equal"] for c in q["cases"])
q=read(OUT/"v100/model-v4/mtp2-qualification/report.json")
assert q["completed"] and all(c["output_ids_equal"] and c["committed_target_state_bytes_equal"] and c["committed_draft_state_equal_mtp1"] for c in q["cases"])
c=read(OUT/"v100/campaign-v5/campaign-report.json")
assert c["completed"] and not any(c["path_promotions"].values())
for p in read(OUT/"trace-correlation-audit.json").values():
    assert all(v["unattributed_graph_nodes_in_ranges"]==0 for v in p["graph_activity"].values())
control=read(ROOT/"benchmarks/v100-batched-mtp/v100/model-v2/build-metadata/build.json")
assert all(sha(ROOT/"src/venturi/ops/csrc"/rel)==v for rel,v in control["source_sha256"].items())
files=[]
for folder in ["src","tests","tools","scripts","docs"]:
    files += [p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"]
files += [ROOT/p for p in ["README.md","pyproject.toml","AGENTS.md"]]
files += [OUT/p for p in ["summary.md","design.md","next-bottlenecks.md","summarize.py","freeze.py"]]
m={}
for p in sorted(set(files)):
    rel=p.relative_to(ROOT);dst=OUT/"source"/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);m[str(rel)]=sha(p)
(OUT/"source-sha256.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
checks=dict(frozen=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(m),prior_capsules_verified=34,native_library_sha256=h,correctness_qualified=True,graph_nodes_actually_replayed=True,unprofiled_screen_passed=False,promoted_to_model_default=False,all_nine_default_native_sources_restored=True,full_external_speed_goal_complete=False,achieved_occupancy=None,all_owned_processes_terminal=True,original_gpu_footprint_restored=True,no_local_rust_or_cuda_compilation=True,no_commit_or_push=True)
(OUT/"evidence-check.json").write_text(json.dumps(checks,indent=2)+"\n")
e={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and "source" not in p.relative_to(OUT).parts and p.name!="evidence-sha256.json" and "__pycache__" not in p.parts and p.suffix!=".pyc"}
(OUT/"evidence-sha256.json").write_text(json.dumps(e,indent=2,sort_keys=True)+"\n")
assert all(sha(OUT/p)==v for p,v in e.items());assert all(sha(OUT/"source"/p)==v for p,v in m.items())
print(json.dumps(dict(frozen=True,source_files=len(m),evidence_files=len(e),prior_capsules_verified=34,goal_complete=False)))
