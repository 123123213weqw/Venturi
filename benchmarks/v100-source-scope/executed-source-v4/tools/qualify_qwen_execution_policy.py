"""Exact trained trajectory/state gate for execution-only source-check policies.

Independent represented-Q8 teacher/draft equations are explicitly reused from
hash-pinned qualified references. This is not a new arithmetic/corpus quality gate.
"""

import argparse
from contextlib import ExitStack
import ctypes as C
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from qualify_qwen_mtp_batch import draft_signature
from qualify_qwen_projection_variant import state_signature
from venturi.models import (
    NativeQwenModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def read_pin(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Execution-policy qualification reference hash mismatch")
    result = json.loads(raw)
    if not isinstance(result, dict) or not result.get("completed"):
        raise ValueError("Execution-policy qualification reference incomplete")
    return result


def forbidden(*args, **kwargs):
    raise AssertionError("Python numerical fallback in native policy qualification")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--cases", required=True)
    for name in ("target", "draft", "batch"):
        p.add_argument("--prior-" + name + "-report", required=True)
        p.add_argument("--prior-" + name + "-report-sha256", required=True)
    p.add_argument("--expected-source-scope", required=True, type=int, choices=(0, 1))
    p.add_argument("--output-directory", required=True)
    a = p.parse_args()
    refs = {
        name: read_pin(
            getattr(a, "prior_" + name + "_report"),
            getattr(a, "prior_" + name + "_report_sha256"),
        )
        for name in ("target", "draft", "batch")
    }
    target_pin = refs["target"]["profiles"]["q8-residual-column-mlp"]["candidate"]
    assert (
        target_pin["passed"]
        and sum(len(c["all_logit_bytes"]) for c in target_pin["cases"]) == 98
    )
    draft_pin = refs["draft"]
    assert (
        draft_pin["projection_profile"] == "q8-residual-column-mlp"
        and draft_pin["mtp_draft_projection_profile"] == "q8-residual"
    )
    assert (
        len(draft_pin["block_reference"]) == 2
        and len(draft_pin["recurrence_reference"]) == 3
    )
    assert all(
        c["passed"]
        for c in draft_pin["block_reference"] + draft_pin["recurrence_reference"]
    )
    assert refs["batch"]["profiles"] == dict(
        target="q8-residual-column-mlp", draft="q8-residual", kv="fp16"
    )
    assert refs["batch"]["selective_reset_untouched_state_exact"]
    cases = json.loads(Path(a.cases).read_text())
    assert len(cases) == 4
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(
        scope=__doc__,
        completed=False,
        cases=[],
        source_scope_mode=a.expected_source_scope,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        reused_reference_sha256={
            name: getattr(a, "prior_" + name + "_report_sha256") for name in refs
        },
        independent_numerical_quality_reused=True,
        new_cpu_teacher_campaign=False,
        broad_bf16_corpus_quality=False,
        projection_rows=32,
        capacity_per_slot=256,
        profiles=dict(target="q8-residual-column-mlp", draft="q8-residual", kv="fp16"),
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(4)
    backend = CudaBackend(a.library)
    backend._native.venturi_qwen_source_scope_mode.argtypes = []
    backend._native.venturi_qwen_source_scope_mode.restype = C.c_int
    assert backend._native.venturi_qwen_source_scope_mode() == a.expected_source_scope
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    assert (
        weights.artifact.artifact_id
        == refs["target"]["artifact_id"]
        == draft_pin["target_artifact_id"]
    )
    report["target_artifact_id"] = weights.artifact.artifact_id
    report["mtp_artifact_sha256"] = hashlib.sha256(
        Path(a.mtp_artifact).read_bytes()
    ).hexdigest()
    assert report["mtp_artifact_sha256"] == draft_pin["mtp_artifact_sha256"]
    target = NativeQwenModel.from_q8(
        weights,
        backend,
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile="q8-residual-column-mlp",
    )
    model = None
    report["source_validation_policy"] = target.projection_info[
        "source_validation_policy"
    ]
    try:
        model = NativeQwenMTPModel.from_q8(
            target,
            QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact)),
            draft_projection_profile="q8-residual",
        )
        for case in cases:
            ordinary = target.create_session(256)
            session = model.create_session(256)
            try:
                session.prepare_prefix()
                kwargs = dict(
                    max_new_tokens=case["max_new_tokens"],
                    eos_token_ids=case.get("eos_token_ids", []),
                )
                with ExitStack() as stack:
                    for name in (
                        "matmul",
                        "mm",
                        "mv",
                        "empty",
                        "zeros",
                        "tensor",
                        "as_tensor",
                    ):
                        stack.enter_context(patch.object(torch, name, forbidden))
                    plain = ordinary.generate(case["prompt_token_ids"], **kwargs)
                    ids = session.generate(
                        case["prompt_token_ids"],
                        draft_tokens=2,
                        recovery="prefix",
                        **kwargs,
                    )
                t = next(c for c in target_pin["cases"] if c["name"] == case["name"])
                d = next(
                    c
                    for c in refs["batch"]["full_references"]
                    if c["name"] == case["name"]
                )
                old = next(
                    c for c in draft_pin["cases"] if c["case"]["name"] == case["name"]
                )
                assert (
                    ids
                    == plain
                    == t["generated_ids"]
                    == d["ids"]
                    == old["generated_ids"]
                )
                state = state_signature(session.snapshot())
                assert (
                    state
                    == state_signature(ordinary.snapshot())
                    == t["committed_private_state"]
                    == d["target_state"]
                )
                draft_state = draft_signature(session.draft_snapshot())
                assert draft_state == d["draft_state"]
                stats = session.stats()
                keys = (
                    "proposed_tokens",
                    "accepted_tokens",
                    "verification_calls",
                    "target_forward_calls",
                    "target_input_tokens",
                    "draft_forward_calls",
                    "draft_input_tokens",
                    "replayed_input_tokens",
                    "forced_proposals",
                )
                assert all(stats[k] == old["mtp_stats"][k] for k in keys)
                report["cases"].append(
                    dict(
                        name=case["name"],
                        generated_ids=ids,
                        position=session.position,
                        exact_ordinary_target_and_pinned_state=True,
                        exact_pinned_corrected_draft_state=True,
                        exact_prior_mtp2_algorithm_counters=True,
                        native_without_python_math=True,
                        target_state=state,
                        draft_state=draft_state,
                        mtp_stats=stats,
                    )
                )
                save()
            finally:
                session.close()
                ordinary.close()
        report["completed"] = True
        save()
    finally:
        if model is not None:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
