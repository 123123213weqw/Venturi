"""Untimed trained graph/eager full verification-byte and committed-state gate."""

import argparse
import hashlib
import json
from pathlib import Path
import torch
from venturi.models import (
    NativeQwenModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend
from qualify_qwen_projection_variant import state_signature, tensor_signature


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draft_signature(state):
    return dict(
        position=state["position"],
        keys=tensor_signature(state["keys"]),
        values=tensor_signature(state["values"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--prior-target-report", required=True)
    p.add_argument("--prior-target-report-sha256", required=True)
    p.add_argument("--output-directory", required=True)
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert digest(a.prior_target_report) == a.prior_target_report_sha256
    assert (
        digest(a.cases)
        == "fd5f82c3a4b4d22dfa05d577ba50c67019b47646440a54616e565d7bdf008d0b"
    )
    prior = json.loads(Path(a.prior_target_report).read_text())
    assert prior["completed"]
    cases = json.loads(Path(a.cases).read_text())
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    assert prior["artifact_id"] == weights.artifact.artifact_id
    target = NativeQwenModel.from_q8(
        weights,
        CudaBackend(a.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile="q8-residual",
    )
    model = None
    report = dict(
        scope=__doc__,
        completed=False,
        profiled=False,
        performance_qualified=False,
        library_sha256=digest(a.library),
        artifact_id=weights.artifact.artifact_id,
        prior_target_report_sha256=a.prior_target_report_sha256,
        cases=[],
        reset_repeat=None,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        model = NativeQwenMTPModel.from_q8(
            target, QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact))
        )
        for case in cases:
            capacity = len(case["prompt_token_ids"]) + case["max_new_tokens"]
            ref, one, plain, graph = (
                target.create_session(capacity),
                model.create_session(capacity),
                model.create_session(capacity),
                model.create_session(capacity),
            )
            try:
                graph.enable_target_layer_graph()
                plain._collect_verification_trace()
                graph._collect_verification_trace()
                args = dict(
                    max_new_tokens=case["max_new_tokens"],
                    eos_token_ids=case.get("eos_token_ids", []),
                )
                expected = ref.generate(case["prompt_token_ids"], **args)
                pinned = next(
                    c
                    for c in prior["profiles"]["q8-residual"]["candidate"]["cases"]
                    if c["name"] == case["name"]
                )
                assert (
                    expected == pinned["generated_ids"]
                    and state_signature(ref.snapshot())
                    == pinned["committed_private_state"]
                )
                assert one.generate(case["prompt_token_ids"], **args) == expected
                for s in [plain, graph]:
                    assert (
                        s.generate(
                            case["prompt_token_ids"],
                            draft_tokens=2,
                            recovery="prefix",
                            **args,
                        )
                        == expected
                    )
                p0, g0 = plain._verification_trace(), graph._verification_trace()
                assert p0 == g0, "Full trained verification head bytes differ"
                target_state = state_signature(ref.snapshot())
                assert (
                    state_signature(plain.snapshot())
                    == state_signature(graph.snapshot())
                    == target_state
                )
                draft = draft_signature(one.draft_snapshot())
                assert (
                    draft_signature(plain.draft_snapshot())
                    == draft_signature(graph.draft_snapshot())
                    == draft
                )
                info = graph.target_layer_graph_info()
                if len(expected) >= 8:
                    assert info["ready"] and info["replays"] > 0
                record = dict(
                    name=case["name"],
                    generated_ids=expected,
                    output_ids_equal=True,
                    all_verification_fp16_bytes_equal=True,
                    verification_widths=p0["widths"],
                    verification_bytes=len(p0["fp16_bytes"]),
                    verification_sha256=hashlib.sha256(p0["fp16_bytes"]).hexdigest(),
                    committed_target_state_equal=True,
                    committed_draft_state_equal=True,
                    target_state=target_state,
                    draft_state=draft,
                    graph=info,
                    stats=graph.stats(),
                )
                report["cases"].append(record)
                save()
                if report["reset_repeat"] is None and info["replays"] > 0:
                    before = info["captures"]
                    for s in [plain, graph]:
                        s.reset()
                    assert (
                        plain.generate(
                            case["prompt_token_ids"],
                            draft_tokens=2,
                            recovery="prefix",
                            **args,
                        )
                        == expected
                    )
                    assert (
                        graph.generate(
                            case["prompt_token_ids"],
                            draft_tokens=2,
                            recovery="prefix",
                            **args,
                        )
                        == expected
                    )
                    assert plain._verification_trace() == graph._verification_trace()
                    assert (
                        state_signature(graph.snapshot()) == target_state
                        and draft_signature(graph.draft_snapshot()) == draft
                    )
                    assert graph.target_layer_graph_info()["captures"] == before
                    report["reset_repeat"] = dict(
                        case=case["name"],
                        same_full_verification_bytes=True,
                        same_target_and_draft_state=True,
                        executable_reused=True,
                    )
                    save()
            finally:
                for s in [ref, one, plain, graph]:
                    s.close()
        assert len(report["cases"]) == 4 and report["reset_repeat"]
        report["completed"] = True
        save()
    finally:
        if model:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
