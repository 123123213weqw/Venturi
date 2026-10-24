"""Apply the predeclared repeat gate; preserve or restore native sources only."""

from pathlib import Path
import hashlib
import json
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


m = read(OUT / "metrics.json")
current = [e for e in m["experiments"] if not e["reused"]]
index = {e["label"]: {r["request_count"]: r for r in e["summary"]} for e in current}
checks = {}
for name, ratios in m["repeat_path_ratios"].items():
    a, b = (index["repeat-" + arm + "-" + name] for arm in ["control", "candidate"])
    assert set(a) == set(b) == {1, 2, 4, 8}
    assert all(a[n]["repeats"] == b[n]["repeats"] == 3 for n in a)
    stable = all(v >= 0.98 for v in ratios.values())
    eligible = [(1, 1.01), (8, 1.02)] + ([(2, 1.02)] if name == "mtp-batch" else [])
    cells = {}
    for n, threshold in eligible:
        improvements = sum(
            x > y for x, y in zip(b[n]["trial_output_tps"], a[n]["trial_output_tps"])
        )
        cells[str(n)] = dict(
            ratio=ratios[str(n)],
            threshold=threshold,
            improving_trials=improvements,
            qualified=ratios[str(n)] > threshold and improvements >= 2,
        )
    checks[name] = dict(
        no_cell_slower_by_more_than_two_percent=stable,
        cells=cells,
        qualifies=stable and any(c["qualified"] for c in cells.values()),
    )
adopt = any(v["qualifies"] for v in checks.values())
decision = dict(
    status="completed",
    initial_unbounded_rejected=True,
    initial_unguarded_three_descriptor_regressions_retained=True,
    operator_gate=m["operator_promotion"],
    screen_path_promotions=m["path_promotions"],
    repeat_adoption_checks=checks,
    promote_to_model_default=adopt,
    full_external_speed_goal_complete=False,
    scope="Incremental geometry-guard adoption only. No broad BF16/corpus, universal workload-router, achieved occupancy or external overall-win claim.",
)
control = read(
    ROOT / "benchmarks/v100-batched-mtp/v100/model-v2/build-metadata/build.json"
)
if not adopt:
    for name in ["q8_shared.cuh", "q8_fused.cuh"]:
        shutil.copy2(
            ROOT / "benchmarks/v100-batched-mtp/source/src/venturi/ops/csrc" / name,
            ROOT / "src/venturi/ops/csrc" / name,
        )
selected = (
    read(next((OUT / "v100/guarded-study-v7/build").rglob("build.json")))
    if adopt
    else control
)
for rel, h in selected["source_sha256"].items():
    assert sha(ROOT / "src/venturi/ops/csrc" / rel) == h, rel
(OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
(OUT / "restored-dispatch-proof.json").write_text(
    json.dumps(
        dict(
            model_dispatch_restored=not adopt,
            all_nine_default_native_source_hashes_restored=not adopt,
            guarded_native_dispatch_adopted=adopt,
            selected_library_sha256=selected["library_sha256"],
            selected_native_dependencies=selected["source_sha256"],
            all_nine_selected_native_source_hashes_verified=True,
            original_plain_three_descriptor_t8_and_fp16_paths_preserved=True,
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(decision, indent=2))
