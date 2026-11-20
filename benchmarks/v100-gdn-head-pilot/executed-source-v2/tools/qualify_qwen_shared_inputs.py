"""Byte-exact trained qualification of shared projection inputs, not a speed run.

Compare the pinned qualified unshared binary with the candidate within each
arithmetic profile. Reuse prior CPU/quality gates only when all new logits,
outputs and committed private caches are byte-exact to the qualified control.
No fresh BF16/corpus-quality claim and no arithmetic-profile substitution.
"""

import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path

import torch

from qualify_qwen_projection_profile import native_generate
from venturi.models import NativeQwenModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_signature(tensor):
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Hashing requires an explicit contiguous CPU diagnostic copy")
    size = tensor.numel() * tensor.element_size()
    view = (ctypes.c_ubyte * size).from_address(tensor.data_ptr())
    return dict(
        dtype=str(tensor.dtype),
        shape=list(tensor.shape),
        bytes=size,
        sha256=hashlib.sha256(view).hexdigest(),
    )


def state_signature(state):
    return {
        key: [
            None if tensors is None else [tensor_signature(t) for t in tensors]
            for tensors in slots
        ]
        for key, slots in state.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--baseline-library", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--prior-report", required=True)
    parser.add_argument("--prior-report-sha256", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-directory", required=True)
    a = parser.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert digest(a.prior_report) == a.prior_report_sha256, (
        "Qualified prior report changed"
    )
    prior = json.loads(Path(a.prior_report).read_text())
    assert prior["completed"] and digest(a.baseline_library) == prior["library_sha256"]
    assert all(p["passed"] for p in prior["profiles"].values())
    cases = json.loads(Path(a.cases).read_text())
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    assert weights.artifact.artifact_id == prior["artifact_id"]
    report = dict(
        scope=__doc__,
        artifact_id=weights.artifact.artifact_id,
        artifact_hash_scope="Prior frozen original-source artifact validation reused, not a fresh payload SHA256 scan; same artifact is loaded by both controls.",
        prior_report_sha256=a.prior_report_sha256,
        baseline_library_sha256=digest(a.baseline_library),
        candidate_library_sha256=digest(a.library),
        profiles={},
        broad_bf16_corpus_quality_qualified=False,
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for profile in ("fp16", "q8-residual"):
        controls = {}
        report["profiles"][profile] = controls
        for label, library in (("unshared", a.baseline_library), ("shared", a.library)):
            model = NativeQwenModel.from_q8(
                weights,
                CudaBackend(library),
                weight_mode="resident",
                dtype=torch.float16,
                max_tokens=32,
                tensor_parallel_device="cuda:1",
                tensor_parallel_transport="host",
                projection_profile=profile,
            )
            record = dict(cases=[])
            controls[label] = record
            try:
                for i, case in enumerate(cases):
                    expected = prior["profiles"][profile]["trained_cases"][i][
                        "generated_ids"
                    ]
                    assert (
                        case["name"]
                        == prior["profiles"][profile]["trained_cases"][i]["name"]
                    )
                    session = model.create_session(256)
                    try:
                        hashes, ids = [], []
                        for step, token in enumerate(expected):
                            prefix = (
                                case["prompt_token_ids"]
                                if step == 0
                                else [expected[step - 1]]
                            )
                            logits = session.forward(prefix)
                            hashes.append(tensor_signature(logits))
                            ids.append(int(logits.argmax()))
                            assert ids[-1] == token, (
                                "Pinned original trajectory changed"
                            )
                        diagnostic = state_signature(session.snapshot())
                        session.reset()
                        generated = native_generate(
                            session,
                            case["prompt_token_ids"],
                            max_new_tokens=case["max_new_tokens"],
                            eos_token_ids=case.get("eos_token_ids", []),
                        )
                        assert generated == ids == expected
                        assert state_signature(session.snapshot()) == diagnostic
                        item = dict(
                            name=case["name"],
                            generated_ids=generated,
                            all_logit_bytes=hashes,
                            committed_private_state=diagnostic,
                            committed_position=session.position,
                            native_generate_without_torch_math=True,
                            diagnostic_generate_state_bytes_equal=True,
                        )
                        record["cases"].append(item)
                        if label == "shared":
                            assert item == controls["unshared"]["cases"][i], (
                                "Candidate changed same-profile logits/cache bytes"
                            )
                        save()
                    finally:
                        session.close()
                order = [0, 1, 2, 3, 3, 2, 1, 0]
                prompts = [cases[i]["prompt_token_ids"] for i in order]
                maximum = [cases[i]["max_new_tokens"] for i in order]
                maximum[4] = 13
                batch = model.create_batch([256] * 8)
                try:
                    output = native_generate(
                        batch,
                        prompts,
                        max_new_tokens=maximum,
                        eos_token_ids=[
                            cases[i].get("eos_token_ids", []) for i in order
                        ],
                    )
                    assert output == [
                        record["cases"][i]["generated_ids"][: maximum[j]]
                        for j, i in enumerate(order)
                    ]
                    record["batch"] = dict(
                        generated_ids=output,
                        private_cache=[
                            state_signature(batch.snapshot(i)) for i in range(8)
                        ],
                        positions=[s["position"] for s in batch.info()["requests"]],
                    )
                    batch.reset(slots=[7, 0])
                    logits = batch.forward([[7, 1], [3, 9, 2]], slots=[7, 0])
                    record["reset_continuation"] = dict(
                        logits=tensor_signature(logits),
                        private_cache=[
                            state_signature(batch.snapshot(i)) for i in range(8)
                        ],
                        positions=[s["position"] for s in batch.info()["requests"]],
                    )
                    if label == "shared":
                        assert record["batch"] == controls["unshared"]["batch"], (
                            "Shared batch changed original profile bytes"
                        )
                        assert (
                            record["reset_continuation"]
                            == controls["unshared"]["reset_continuation"]
                        )
                finally:
                    batch.close()
                record["passed"] = True
                save()
            finally:
                model.close()
                gc.collect()
                torch.cuda.empty_cache()
        controls["same_profile_bytes_equal"] = (
            controls["shared"] == controls["unshared"]
        )
        assert controls["same_profile_bytes_equal"]
        save()
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
