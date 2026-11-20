"""Derive opt-in adoption and incomplete-goal decisions from measured gates."""

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROFILE = "q8-residual-column-mlp"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "evidence-sha256.json").exists()
    metrics = read(OUT / "metrics.json")
    qualification = read(OUT / "v100/model-v2/qualification/report.json")
    budget = read(OUT / "v100/budget-v5/qualification/report.json")
    compatibility = read(OUT / "external-control-compatibility.json")
    checks = [
        check
        for case in qualification["profiles"][PROFILE]["candidate"]["cases"]
        for check in case["teacher_logit_checks"]
    ]
    assert len(checks) == 98 and all(c["passed"] for c in checks)
    adopted = (
        qualification["completed"]
        and budget["completed"]
        and budget["same_profile_row_budget_bytes_equal"]
        and compatibility["matched"]
        and all(c["qualifies"] for c in metrics["adoption_checks"].values())
    )
    assert adopted
    decision = dict(
        profile=PROFILE,
        adopt_as_explicit_opt_in=adopted,
        change_existing_defaults=False,
        automatic_strategy_router_implemented=False,
        architecture_scope="Qwen native resident FP16 target/KV, SM70 two-GPU explicit host TP; MLP column/row only. Original row-TP attention/GDN and FP16 draft retained.",
        arithmetic_change="Stored Q8 codes/scales preserved without requantization; FP32 partial transport/sum before a single FP16 cast. FP32 association differs from prior full-K down; not old-profile byte parity.",
        quality_scope="98 represented-Q8/FP16-teacher logit rows plus four final private-state trajectories, batch/reset and MTP gates; not an original-BF16 model or corpus-quality evaluation.",
        teacher_logit_rows=len(checks),
        maximum_relative_l2=max(c["relative_l2"] for c in checks),
        minimum_centered_cosine=min(c["centered_cosine"] for c in checks),
        maximum_reference_kl=max(c["reference_kl_to_native"] for c in checks),
        all_top1_equal=all(c["native_top1"] == c["reference_top1"] for c in checks),
        adoption_checks=metrics["adoption_checks"],
        best_native_http_by_count=metrics["best_repeated_native_http_by_count"],
        stronger_external_http_by_count=metrics[
            "best_known_pinned_external_http_by_count"
        ],
        complete_http_matrix_goal_met=metrics["complete_http_matrix_goal_met"],
        single_decode_comparison=metrics["single_decode_comparison"],
        full_external_speed_goal_complete=metrics["external_speed_goal_complete"],
        achieved_occupancy=None,
        counter_policy_changed=False,
        metrics_sha256=sha(OUT / "metrics.json"),
        qualification_report_sha256=sha(
            OUT / "v100/model-v2/qualification/report.json"
        ),
        row_budget_qualification_sha256=sha(
            OUT / "v100/budget-v5/qualification/report.json"
        ),
        external_control_compatibility_sha256=sha(
            OUT / "external-control-compatibility.json"
        ),
    )
    assert decision["complete_http_matrix_goal_met"]
    assert not decision["full_external_speed_goal_complete"]
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("Adopt explicit new profile; full external speed goal remains incomplete")


if __name__ == "__main__":
    main()
