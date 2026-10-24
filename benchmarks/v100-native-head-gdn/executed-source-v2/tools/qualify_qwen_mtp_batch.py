"""Fresh trained MTP-batch state/identity gates with pinned target quality reuse."""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
from qualify_qwen_projection_variant import state_signature, tensor_signature
from venturi.models import (
    NativeQwenModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def draft_signature(s):
    return dict(
        position=s["position"],
        keys=tensor_signature(s["keys"]),
        values=tensor_signature(s["values"]),
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
    p.add_argument(
        "--projection-profile",
        choices=("q8-residual", "q8-residual-column-mlp"),
        default="q8-residual",
    )
    p.add_argument(
        "--mtp-draft-projection-profile",
        choices=("fp16", "q8-residual"),
        default="fp16",
    )
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert sha(a.prior_target_report) == a.prior_target_report_sha256
    prior = json.loads(Path(a.prior_target_report).read_text())
    assert prior["completed"]
    pinned = prior["profiles"][a.projection_profile]["candidate"]
    assert sum(len(c["all_logit_bytes"]) for c in pinned["cases"]) == 98
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    cases = json.loads(Path(a.cases).read_text())
    assert len(cases) == 4
    report = dict(
        scope=__doc__,
        completed=False,
        library_sha256=sha(a.library),
        target_artifact_id=weights.artifact.artifact_id,
        mtp_artifact_sha256=sha(a.mtp_artifact),
        prior_target_report_sha256=a.prior_target_report_sha256,
        projection_rows=32,
        capacity_per_slot=256,
        profiles=dict(
            target=a.projection_profile, draft=a.mtp_draft_projection_profile, kv="fp16"
        ),
        new_cpu_teacher_campaign=False,
        broad_bf16_corpus_quality=False,
        full_references=[],
        cohorts=[],
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    target = NativeQwenModel.from_q8(
        weights,
        CudaBackend(a.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile=a.projection_profile,
    )
    model = batch = None

    def reference(case, budget):
        r = target.create_session(256)
        s = model.create_session(256)
        try:
            args = dict(
                max_new_tokens=budget, eos_token_ids=case.get("eos_token_ids", [])
            )
            ids = r.generate(case["prompt_token_ids"], **args)
            assert s.generate(case["prompt_token_ids"], **args) == ids
            state = state_signature(r.snapshot())
            assert state_signature(s.snapshot()) == state
            return dict(
                ids=ids,
                target_state=state,
                draft_state=draft_signature(s.draft_snapshot()),
            )
        finally:
            r.close()
            s.close()
            gc.collect()

    def run(entries, slots):
        batch.reset()
        prompts = [c["prompt_token_ids"] for c, n in entries]
        budgets = [n for c, n in entries]
        eos = [c.get("eos_token_ids", []) for c, n in entries]
        refs = [reference(c, n) for c, n in entries]

        def forbidden(*a, **kw):
            raise AssertionError("Python numerical fallback in native MTP batch")

        with ExitStack() as stack:
            for name in [
                "empty",
                "zeros",
                "matmul",
                "mm",
                "softmax",
                "sigmoid",
                "cat",
                "tensor",
            ]:
                stack.enter_context(patch.object(torch, name, forbidden))
            ids = batch.generate(
                prompts, max_new_tokens=budgets, eos_token_ids=eos, slots=slots
            )
        assert ids == [r["ids"] for r in refs]
        states, drafts = [], []
        for slot, ref in zip(slots, refs):
            state = state_signature(batch.snapshot(slot))
            draft = draft_signature(batch.draft_snapshot(slot))
            assert state == ref["target_state"] and draft == ref["draft_state"]
            states.append(state)
            drafts.append(draft)
        info = batch.info()
        assert all(
            info["requests"][i]["mtp"]["forced_proposals"]
            == info["requests"][i]["mtp"]["replayed_input_tokens"]
            == 0
            for i in slots
        )
        record = dict(
            slots=slots,
            budgets=budgets,
            generated_ids=ids,
            exact_target_state=True,
            exact_corrected_draft_state=True,
            native_without_python_math=True,
            target_states=states,
            draft_states=drafts,
            info=info,
        )
        report["cohorts"].append(record)
        save()

    try:
        model = NativeQwenMTPModel.from_q8(
            target,
            QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact)),
            draft_projection_profile=a.mtp_draft_projection_profile,
        )
        batch = model.create_batch([256] * 8)
        for case in cases:
            ref = reference(case, case["max_new_tokens"])
            old = next(c for c in pinned["cases"] if c["name"] == case["name"])
            assert (
                ref["ids"] == old["generated_ids"]
                and ref["target_state"] == old["committed_private_state"]
            )
            report["full_references"].append(
                dict(name=case["name"], **ref, pinned_prior_final_state_equal=True)
            )
            save()
        entries = [
            (
                cases[i % 4],
                cases[i % 4]["max_new_tokens"] if i < 4 else [1, 2, 3, 7][i % 4],
            )
            for i in range(8)
        ]
        run(entries, list(reversed(range(8))))
        # Reset/regenerate only selected slots and prove untouched target/draft bytes.
        before = [state_signature(batch.snapshot(i)) for i in range(8)]
        before_draft = [draft_signature(batch.draft_snapshot(i)) for i in range(8)]
        batch.reset(slots=[7, 0])
        entries2 = [(cases[2], 5), (cases[0], 2)]
        refs = [reference(c, n) for c, n in entries2]
        assert batch.generate(
            [c["prompt_token_ids"] for c, n in entries2],
            max_new_tokens=[n for c, n in entries2],
            eos_token_ids=[c.get("eos_token_ids", []) for c, n in entries2],
            slots=[7, 0],
        ) == [r["ids"] for r in refs]
        for i in range(1, 7):
            assert (
                state_signature(batch.snapshot(i)) == before[i]
                and draft_signature(batch.draft_snapshot(i)) == before_draft[i]
            )
        for i, r in zip([7, 0], refs):
            assert (
                state_signature(batch.snapshot(i)) == r["target_state"]
                and draft_signature(batch.draft_snapshot(i)) == r["draft_state"]
            )
        report["selective_reset_untouched_state_exact"] = True
        run([(cases[0], 32)] * 8, list(range(8)))
        report["completed"] = True
        save()
    finally:
        if batch is not None:
            batch.close()
        if model is not None:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
