"""New distributed-profile teacher/trajectory/private-cache qualification.

Reuse immutable original-FP16 teacher rows with verified hashes, not a new
unchanged-FP16 campaign. The candidate must pass the same represented-weight
thresholds and complete trajectories; this is not original BF16 corpus quality.
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

PROFILE = "q8-residual-column-mlp-head-gdn"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--teacher-report", required=True)
    p.add_argument("--teacher-report-sha256", required=True)
    p.add_argument("--projection-rows", type=int, choices=(32, 256), default=32)
    p.add_argument("--output-directory", required=True)
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert sha(a.teacher_report) == a.teacher_report_sha256
    prior = json.loads(Path(a.teacher_report).read_text())
    assert prior["completed"]
    reference = prior["profiles"]["fp16"]["candidate"]
    assert (
        reference["passed"]
        and sum(len(c["all_logit_bytes"]) for c in reference["cases"]) == 98
    )
    teachers = {}
    for name, pin in prior["teacher_row_files"].items():
        path = Path(pin.get("source_path", str(Path(a.teacher_report).parent / name)))
        assert path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"]
        teachers[name] = path
    cases = json.loads(Path(a.cases).read_text())
    assert len(cases) == len(reference["cases"]) == 4
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    assert weights.artifact.artifact_id == prior["artifact_id"]
    report = dict(
        scope=__doc__,
        completed=False,
        artifact_id=weights.artifact.artifact_id,
        library_sha256=sha(a.library),
        teacher_report_sha256=a.teacher_report_sha256,
        cases_sha256=sha(a.cases),
        teacher_rows_reused=98,
        new_unchanged_fp16_campaign=False,
        original_bf16_corpus_claim=False,
        fixed_logit_thresholds=dict(
            relative_l2=0.02,
            centered_cosine=0.999,
            reference_kl_to_native=0.002,
            top1_equal=True,
        ),
        fixed_work=dict(
            max_tokens=a.projection_rows, capacity=256, kv="fp16", transport="host"
        ),
        profiles={
            PROFILE: dict(candidate=dict(cases=[]), changed_fp32_association=True)
        },
        teacher_row_files={
            name: dict(
                source_path=str(path), sha256=sha(path), bytes=path.stat().st_size
            )
            for name, path in teachers.items()
        },
    )
    record = report["profiles"][PROFILE]["candidate"]

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    model = NativeQwenModel.from_q8(
        weights,
        CudaBackend(a.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        max_tokens=a.projection_rows,
        projection_profile=PROFILE,
    )
    record["projection_info"] = model.projection_info
    try:
        for i, case in enumerate(cases):
            expected = reference["cases"][i]["generated_ids"]
            assert case["name"] == reference["cases"][i]["name"]
            teacher = torch.load(
                teachers[f"teacher-{i}.pt"], map_location="cpu", weights_only=True
            )
            assert teacher.shape[0] == len(expected)
            r = model.create_session(256)
            try:
                ids, hashes, checks = [], [], []
                for step, wanted in enumerate(expected):
                    logits = r.forward(
                        case["prompt_token_ids"] if step == 0 else [expected[step - 1]]
                    )
                    check = logits_comparison(logits, teacher[step])
                    assert check["passed"], (
                        "Unchanged archived teacher quality gate failed"
                    )
                    ids.append(int(logits.argmax()))
                    assert ids[-1] == wanted, "Original teacher trajectory changed"
                    hashes.append(tensor_signature(logits))
                    checks.append(check)
                state = state_signature(r.snapshot())
                position = r.position
                r.reset()
                generated = native_generate(
                    r,
                    case["prompt_token_ids"],
                    max_new_tokens=case["max_new_tokens"],
                    eos_token_ids=case.get("eos_token_ids", []),
                )
                assert (
                    generated == ids == expected
                    and state_signature(r.snapshot()) == state
                )
                record["cases"].append(
                    dict(
                        name=case["name"],
                        generated_ids=ids,
                        all_logit_bytes=hashes,
                        teacher_logit_checks=checks,
                        committed_private_state=state,
                        committed_position=position,
                        native_generate_without_torch_math=True,
                        diagnostic_generate_state_bytes_equal=True,
                        info=r.info(),
                    )
                )
                save()
            finally:
                r.close()
                del teacher
                gc.collect()
        order = [0, 1, 2, 3, 3, 2, 1, 0]
        prompts = [cases[i]["prompt_token_ids"] for i in order]
        budgets = [cases[i]["max_new_tokens"] for i in order]
        budgets[4] = 13
        batch = model.create_batch([256] * 8)
        try:
            generated = native_generate(
                batch,
                prompts,
                max_new_tokens=budgets,
                eos_token_ids=[cases[i].get("eos_token_ids", []) for i in order],
            )
            assert generated == [
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
                            eos_token_ids=cases[order[slot]].get("eos_token_ids", []),
                        )
                        == generated[slot]
                    )
                    assert state_signature(r.snapshot()) == state_signature(
                        batch.snapshot(slot)
                    )
                finally:
                    r.close()
            before = [state_signature(batch.snapshot(i)) for i in range(8)]
            record["batch"] = dict(
                generated_ids=generated,
                private_cache=before,
                positions=[r["position"] for r in batch.info()["requests"]],
                info=batch.info(),
            )
            batch.reset(slots=[7, 0])
            new = [[7, 1], [3, 9, 2]]
            logits = batch.forward(new, slots=[7, 0])
            after = [state_signature(batch.snapshot(i)) for i in range(8)]
            assert all(before[i] == after[i] for i in range(1, 7))
            for row, slot in enumerate([7, 0]):
                r = model.create_session(256)
                try:
                    assert torch.equal(
                        logits[row].view(torch.uint8),
                        r.forward(new[row]).view(torch.uint8),
                    )
                    assert state_signature(r.snapshot()) == after[slot]
                finally:
                    r.close()
            record["reset_continuation"] = dict(
                logits=tensor_signature(logits),
                private_cache=after,
                positions=[r["position"] for r in batch.info()["requests"]],
            )
            record["passed"] = True
            report["completed"] = True
            save()
        finally:
            batch.close()
    finally:
        model.close()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
