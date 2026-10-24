"""Trained-model qualification of explicit residual activation arithmetic.

Fixed prior CPU-logit thresholds are unchanged. Original FP16 byte gates remain
separate. Additional teacher forcing follows the original greedy trajectory;
same-profile scalar/batch cache bytes must agree. No throughput or corpus claim.
"""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_qwen_tensor_parallel import same_state  # noqa: E402
from qualify_qwen_native import compare_saved  # noqa: E402
from qualify_qwen_streaming import logits_comparison  # noqa: E402
from venturi.models import NativeQwenModel, QwenQ8Weights  # noqa: E402
from venturi.ninfer import NInferArtifact  # noqa: E402
from venturi.ops import CudaBackend  # noqa: E402


CPU_REPORT_SHA = "cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732"
SNAPSHOT_SHA = {
    "native-step-0.pt": "5f02be25b4642721c8a66e8f4be57e3a370fe7b43e1185f4e260a9ee5d189926",
    "reference-step-0.pt": "b151a8c0a7677a83dcd72f42898d9a0889f9335c50593d2444454160f354cddd",
    "native-step-1.pt": "73b69c734d60b16da2d6a3a3cf43c9a36b8634e21ac3130a83051b0a16139e80",
    "reference-step-1.pt": "a4f3216ef49cdb4d61d8fe5d6ee2b404ca28de6733159acf9d951b441522518f",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def native_generate(session, prompts, **options):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Torch numerical operation/allocation during native generation"
        )

    with ExitStack() as stack:
        for name in (
            "empty",
            "zeros",
            "mm",
            "matmul",
            "bmm",
            "argmax",
            "tensor",
            "cat",
            "softmax",
            "sigmoid",
        ):
            stack.enter_context(patch.object(torch, name, forbidden))
        return session.generate(prompts, **options)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--cpu-reference-directory", required=True)
    p.add_argument("--output-directory", required=True)
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    reference_dir = Path(a.cpu_reference_directory)
    assert digest(reference_dir / "report.json") == CPU_REPORT_SHA, (
        "Pinned CPU report changed"
    )
    original = json.loads((reference_dir / "report.json").read_text())
    assert (
        original["teacher_forced_cpu_reference_completed"]
        and original["numerical_agreement_gate_passed"]
    )
    assert digest(a.artifact) == original["artifact_sha256"], (
        "Official-source Q8 artifact changed"
    )
    for name, sha in SNAPSHOT_SHA.items():
        assert digest(reference_dir / name) == sha, "Pinned CPU/native snapshot changed"
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    assert weights.artifact.artifact_id == "3d635b87db004f42aefd6730e372e530"
    cases = json.loads(Path(a.cases).read_text())
    report = dict(
        scope=__doc__,
        library_sha256=digest(a.library),
        artifact_id=weights.artifact.artifact_id,
        cpu_report_sha256=CPU_REPORT_SHA,
        cpu_snapshot_sha256=SNAPSHOT_SHA,
        profiles={},
        fixed_logit_thresholds=dict(
            relative_l2=0.02,
            centered_cosine=0.999,
            reference_kl_to_native=0.002,
            top1_equal=True,
        ),
        full_original_bf16_corpus_quality_qualified=False,
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for profile in ("fp16", "q8-residual"):
        model = NativeQwenModel.from_q8(
            weights,
            CudaBackend(a.library),
            weight_mode="resident",
            dtype=torch.float16,
            device="cuda:0",
            max_tokens=32,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            projection_profile=profile,
        )
        record = dict(
            projection_info=model.projection_info,
            fixed_cpu_steps=[],
            trained_cases=[],
            same_profile_batch=None,
        )
        report["profiles"][profile] = record
        save()
        try:
            session = model.create_session(256)
            try:
                for step, token in enumerate(original["native_generated_ids"]):
                    ids = (
                        original["case"]["prompt_token_ids"]
                        if step == 0
                        else [original["native_generated_ids"][step - 1]]
                    )
                    trace = []
                    logits = session.forward(ids, trace=trace)
                    cpu = torch.load(
                        reference_dir / f"reference-step-{step}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    old = torch.load(
                        reference_dir / f"native-step-{step}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    current = dict(logits=logits, trace=trace, state=session.snapshot())
                    comparisons = compare_saved(current, old)
                    item = dict(
                        step=step,
                        expected_id=token,
                        logits_vs_pinned_cpu=logits_comparison(logits, cpu["logits"]),
                        original_native_logit_bytes_equal=torch.equal(
                            logits.view(torch.uint8), old["logits"].view(torch.uint8)
                        ),
                        original_hidden_state_comparisons=comparisons,
                    )
                    record["fixed_cpu_steps"].append(item)
                    save()
                    assert item["logits_vs_pinned_cpu"]["passed"], (
                        "Unchanged trained CPU logit gate failed"
                    )
                    if profile == "fp16":
                        assert item["original_native_logit_bytes_equal"] and all(
                            c["bytes_equal"] for c in comparisons
                        ), "Default original precision/state changed"
                    del current, cpu, old, trace, logits, comparisons
                    gc.collect()
            finally:
                session.close()

            for index, case in enumerate(cases):
                session = model.create_session(256)
                try:
                    reference = (
                        None
                        if profile == "fp16"
                        else torch.load(
                            out / f"original-case-{index}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
                    ids, logits_rows, checks = [], [], []
                    expected = reference["ids"] if reference is not None else None
                    steps = (
                        len(expected)
                        if expected is not None
                        else case["max_new_tokens"]
                    )
                    for step in range(steps):
                        prefix = (
                            case["prompt_token_ids"]
                            if step == 0
                            else [
                                expected[step - 1] if expected is not None else ids[-1]
                            ]
                        )
                        logits = session.forward(prefix)
                        value = int(logits.argmax())
                        ids.append(value)
                        if reference is None:
                            logits_rows.append(logits)
                            if value in case.get("eos_token_ids", []):
                                break
                        else:
                            checks.append(
                                logits_comparison(logits, reference["logits"][step])
                            )
                    if profile == "fp16":
                        if "expected_generated_ids" in case:
                            assert (
                                ids
                                == case["expected_generated_ids"][
                                    : case["max_new_tokens"]
                                ]
                            ), "Original pinned text IDs changed"
                        torch.save(
                            dict(ids=ids, logits=torch.stack(logits_rows)),
                            out / f"original-case-{index}.pt",
                        )
                    item = dict(
                        name=case["name"],
                        teacher_forced_original_ids=ids,
                        original_ids=expected or ids,
                        teacher_forced_logit_checks=checks,
                    )
                    record["trained_cases"].append(item)
                    save()
                    if reference is not None:
                        assert all(c["passed"] for c in checks) and ids == expected, (
                            "Trained original-trajectory logit/top1 gate failed"
                        )
                    state = session.snapshot()
                    session.reset()
                    generated = native_generate(
                        session,
                        case["prompt_token_ids"],
                        max_new_tokens=case["max_new_tokens"],
                        eos_token_ids=case.get("eos_token_ids", []),
                    )
                    assert generated == ids, (
                        "Native generation differs from qualified greedy trajectory"
                    )
                    same_state(session.snapshot(), state)
                    item.update(
                        generated_ids=generated,
                        native_generation_without_torch_math=True,
                        same_profile_diagnostic_generate_state_bytes_exact=True,
                    )
                    save()
                    del state, reference, logits_rows
                    gc.collect()
                finally:
                    session.close()

            if profile == "q8-residual":
                order = [0, 1, 2, 3, 3, 2, 1, 0]
                maximum = [cases[i]["max_new_tokens"] for i in order]
                maximum[4] = 13
                batch = model.create_batch([256] * 8)
                try:
                    prompts = [cases[i]["prompt_token_ids"] for i in order]
                    eos = [cases[i].get("eos_token_ids", []) for i in order]
                    expected = [
                        report["profiles"]["fp16"]["trained_cases"][i]["generated_ids"][
                            : maximum[j]
                        ]
                        for j, i in enumerate(order)
                    ]
                    actual = native_generate(
                        batch, prompts, max_new_tokens=maximum, eos_token_ids=eos
                    )
                    assert actual == expected, (
                        "Heterogeneous residual-profile batch IDs changed"
                    )
                    for j in range(8):
                        scalar = model.create_session(256)
                        try:
                            assert (
                                scalar.generate(
                                    prompts[j],
                                    max_new_tokens=maximum[j],
                                    eos_token_ids=eos[j],
                                )
                                == actual[j]
                            )
                            same_state(scalar.snapshot(), batch.snapshot(j))
                        finally:
                            scalar.close()
                    record["same_profile_batch"] = dict(
                        generated_ids=actual,
                        original_ids_equal=True,
                        all_eight_private_cache_bytes_match_scalar=True,
                        budgets=maximum,
                        info=batch.info(),
                    )
                    save()
                finally:
                    batch.close()
            record["passed"] = True
            save()
        finally:
            model.close()
            gc.collect()
            torch.cuda.empty_cache()
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
