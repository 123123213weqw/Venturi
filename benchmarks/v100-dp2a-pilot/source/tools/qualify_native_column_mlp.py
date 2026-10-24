"""Fresh new-profile teacher, trajectory and private-cache gates, not a speed run.

Original FP16 logits/state are checked against pinned complete signatures first;
new FP32-association arithmetic must meet the unchanged represented-Q8/FP16
teacher quality limits. New-profile scalar/batch/reset states must agree exactly.
This does not establish original BF16 or broad corpus quality.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from qualify_qwen_projection_profile import native_generate
from qualify_qwen_projection_variant import state_signature, tensor_signature
from qualify_qwen_streaming import logits_comparison
from venturi.models import NativeQwenModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend

PROFILE = "q8-residual-column-mlp"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--prior-teacher-report", required=True)
    p.add_argument("--prior-teacher-report-sha256", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--projection-rows", type=int, choices=(32, 256), default=32)
    p.add_argument("--reuse-qualified-report")
    p.add_argument("--reuse-qualified-report-sha256")
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert sha(a.prior_teacher_report) == a.prior_teacher_report_sha256
    prior = json.loads(Path(a.prior_teacher_report).read_text())
    assert prior["completed"]
    reference = prior["profiles"]["fp16"]["candidate"]
    assert (
        reference["passed"]
        and sum(len(c["all_logit_bytes"]) for c in reference["cases"]) == 98
    )
    cases = json.loads(Path(a.cases).read_text())
    assert len(cases) == len(reference["cases"]) == 4
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    assert weights.artifact.artifact_id == prior["artifact_id"]
    qualified = None
    teacher_base = out
    if bool(a.reuse_qualified_report) != bool(a.reuse_qualified_report_sha256):
        p.error("Both qualified-report path and SHA256 are required")
    if a.reuse_qualified_report:
        assert sha(a.reuse_qualified_report) == a.reuse_qualified_report_sha256
        qualified = json.loads(Path(a.reuse_qualified_report).read_text())
        assert qualified["completed"] and qualified["library_sha256"] == sha(a.library)
        assert qualified["artifact_id"] == weights.artifact.artifact_id
        assert qualified["profiles"]["fp16"]["candidate"]["passed"]
        assert qualified["profiles"][PROFILE]["candidate"]["passed"]
        teacher_base = Path(a.reuse_qualified_report).parent
        for name, pin in qualified["teacher_row_files"].items():
            assert sha(teacher_base / name) == pin["sha256"]
    report = dict(
        scope=__doc__,
        completed=False,
        profiled=False,
        artifact_id=weights.artifact.artifact_id,
        library_sha256=sha(a.library),
        prior_teacher_report_sha256=a.prior_teacher_report_sha256,
        artifact_hash_scope="Pinned prior payload/official-source validation reused; current artifact ID matches. No new full payload SHA256 scan or original BF16 corpus claim.",
        cases_sha256=sha(a.cases),
        fixed_logit_thresholds=dict(
            relative_l2=0.02,
            centered_cosine=0.999,
            reference_kl_to_native=0.002,
            top1_equal=True,
        ),
        fixed_work=dict(
            max_tokens=a.projection_rows,
            capacity_per_slot=256,
            dtype="fp16",
            tensor_parallel_transport="host",
            logical_devices=[0, 1],
        ),
        profiles={},
        teacher_row_files={},
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    report["reused_qualified_report_sha256"] = a.reuse_qualified_report_sha256
    if qualified is not None:
        report["teacher_row_files"] = {
            name: dict(**pin, source_path=str(teacher_base / name))
            for name, pin in qualified["teacher_row_files"].items()
        }
    for profile in ["fp16", PROFILE] if qualified is None else [PROFILE]:
        model = NativeQwenModel.from_q8(
            weights,
            CudaBackend(a.library),
            weight_mode="resident",
            dtype=torch.float16,
            device="cuda:0",
            max_tokens=a.projection_rows if profile == PROFILE else 32,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            projection_profile=profile,
        )
        record = dict(cases=[], projection_info=model.projection_info)
        report["profiles"][profile] = dict(
            candidate=record, changed_fp32_association=profile == PROFILE
        )
        try:
            for index, case in enumerate(cases):
                expected = reference["cases"][index]["generated_ids"]
                assert case["name"] == reference["cases"][index]["name"]
                r = model.create_session(256)
                try:
                    ids, hashes, rows, checks = [], [], [], []
                    teacher = (
                        None
                        if profile == "fp16"
                        else torch.load(
                            teacher_base / f"teacher-{index}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
                    for step, wanted in enumerate(expected):
                        prefix = (
                            case["prompt_token_ids"]
                            if step == 0
                            else [expected[step - 1]]
                        )
                        logits = r.forward(prefix)
                        ids.append(int(logits.argmax()))
                        hashes.append(tensor_signature(logits))
                        if profile == "fp16":
                            assert (
                                hashes[-1]
                                == reference["cases"][index]["all_logit_bytes"][step]
                            ), "Default FP16 teacher changed"
                            rows.append(logits)
                        else:
                            check = logits_comparison(logits, teacher[step])
                            checks.append(check)
                            assert check["passed"], (
                                "Unchanged represented teacher quality gate failed"
                            )
                        assert ids[-1] == wanted, "Original teacher trajectory changed"
                    state = state_signature(r.snapshot())
                    if profile == "fp16":
                        assert (
                            state
                            == reference["cases"][index]["committed_private_state"]
                        )
                        file = out / f"teacher-{index}.pt"
                        torch.save(torch.stack(rows), file)
                        report["teacher_row_files"][file.name] = dict(
                            bytes=file.stat().st_size, sha256=sha(file)
                        )
                    r.reset()
                    generated = native_generate(
                        r,
                        case["prompt_token_ids"],
                        max_new_tokens=case["max_new_tokens"],
                        eos_token_ids=case.get("eos_token_ids", []),
                    )
                    assert generated == ids == expected
                    assert state_signature(r.snapshot()) == state, (
                        "Same-profile ordinary diagnostic/generate state changed"
                    )
                    record["cases"].append(
                        dict(
                            name=case["name"],
                            generated_ids=ids,
                            all_logit_bytes=hashes,
                            teacher_logit_checks=checks,
                            committed_private_state=state,
                            committed_position=r.position,
                            native_generate_without_torch_math=True,
                            diagnostic_generate_state_bytes_equal=True,
                        )
                    )
                    if qualified is not None:
                        previous = qualified["profiles"][PROFILE]["candidate"]["cases"][
                            index
                        ]
                        for field in [
                            "generated_ids",
                            "all_logit_bytes",
                            "committed_private_state",
                            "committed_position",
                        ]:
                            assert record["cases"][-1][field] == previous[field], (
                                "Row budget changed new-profile trajectory/state bytes"
                            )
                    save()
                    del rows, teacher
                finally:
                    r.close()
                    gc.collect()
            order = [0, 1, 2, 3, 3, 2, 1, 0]
            prompts = [cases[i]["prompt_token_ids"] for i in order]
            budgets = [cases[i]["max_new_tokens"] for i in order]
            budgets[4] = 13
            batch = model.create_batch([256] * 8)
            try:
                ids = native_generate(
                    batch,
                    prompts,
                    max_new_tokens=budgets,
                    eos_token_ids=[cases[i].get("eos_token_ids", []) for i in order],
                )
                assert ids == [
                    reference["cases"][i]["generated_ids"][:n]
                    for i, n in zip(order, budgets)
                ]
                for slot in range(8):
                    r = model.create_session(256)
                    try:
                        assert (
                            r.generate(
                                prompts[slot],
                                max_new_tokens=budgets[slot],
                                eos_token_ids=cases[order[slot]].get(
                                    "eos_token_ids", []
                                ),
                            )
                            == ids[slot]
                        )
                        assert state_signature(r.snapshot()) == state_signature(
                            batch.snapshot(slot)
                        )
                    finally:
                        r.close()
                record["batch"] = dict(
                    generated_ids=ids,
                    private_cache=[
                        state_signature(batch.snapshot(i)) for i in range(8)
                    ],
                    positions=[s["position"] for s in batch.info()["requests"]],
                )
                if profile == "fp16":
                    assert record["batch"] == reference["batch"]
                if qualified is not None:
                    assert (
                        record["batch"]
                        == qualified["profiles"][PROFILE]["candidate"]["batch"]
                    ), "Row budget changed packed private states"
                before = record["batch"]["private_cache"]
                batch.reset(slots=[7, 0])
                next_ids = [[7, 1], [3, 9, 2]]
                logits = batch.forward(next_ids, slots=[7, 0])
                after = [state_signature(batch.snapshot(i)) for i in range(8)]
                for slot in range(1, 7):
                    assert after[slot] == before[slot]
                for row, slot in enumerate([7, 0]):
                    r = model.create_session(256)
                    try:
                        scalar = r.forward(next_ids[row])
                        assert torch.equal(
                            logits[row].view(torch.uint8), scalar.view(torch.uint8)
                        )
                        assert state_signature(r.snapshot()) == after[slot]
                    finally:
                        r.close()
                record["reset_continuation"] = dict(
                    logits=tensor_signature(logits),
                    private_cache=after,
                    positions=[s["position"] for s in batch.info()["requests"]],
                )
                if profile == "fp16":
                    assert (
                        record["reset_continuation"] == reference["reset_continuation"]
                    )
                if qualified is not None:
                    assert (
                        record["reset_continuation"]
                        == qualified["profiles"][PROFILE]["candidate"][
                            "reset_continuation"
                        ]
                    ), "Row budget changed selective reset/continuation"
                    report["same_profile_row_budget_bytes_equal"] = True
                record["passed"] = True
                save()
            finally:
                batch.close()
        finally:
            model.close()
            gc.collect()
            torch.cuda.empty_cache()
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
