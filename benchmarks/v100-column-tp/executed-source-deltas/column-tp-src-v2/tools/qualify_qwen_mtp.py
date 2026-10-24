"""Trained MTP source-equation and transactional target regression qualification.

CPU reference is represented Q8/FP16, not original-BF16 corpus quality. The MTP
block is checked independently of the target output gate: a broken draft could
otherwise pass greedy target parity with zero acceptance. No performance claims
come from these diagnostic calls or CPU snapshots.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_qwen_mtp import mtp_oracle  # noqa: E402
from test_qwen_tensor_parallel import same_state  # noqa: E402
from qualify_qwen_streaming import metrics  # noqa: E402
from venturi.models import (
    NativeQwenModel,
    NativeQwenPipelineModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)  # noqa: E402
from venturi.ninfer import NInferArtifact  # noqa: E402
from venturi.ops import CudaBackend  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(16 * 2**20), b""):
            h.update(data)
    return h.hexdigest()


def represented_source(value):
    """Independent vectorized FP64 reconstruction, never the export/runtime quantizer."""
    v = value.double()
    groups = v.reshape(v.shape[0], -1, 32)
    scales = (groups.abs().amax(-1) / 127).half().double()
    divisor = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(groups / divisor[:, :, None]).clamp(-127, 127)
    return (codes * scales[:, :, None]).reshape(v.shape)


def block_comparison(actual, reference):
    result = metrics(actual, reference)
    a, b = actual.double().flatten(), reference.double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    denominator = float(ac.norm() * bc.norm())
    result["centered_cosine"] = (
        float(ac @ bc) / denominator if denominator else float(torch.equal(a, b))
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--mode", choices=("tp", "pp"), required=True)
    p.add_argument(
        "--projection-profile", choices=("fp16", "q8-residual"), default="fp16"
    )
    p.add_argument("--output-directory", required=True)
    p.add_argument("--draft-tokens", type=int, choices=(1, 2), default=1)
    p.add_argument("--recovery", choices=("replay", "prefix"), default="replay")
    p.add_argument("--target-layer-graph", action="store_true")
    p.add_argument("--prior-target-report")
    p.add_argument("--prior-target-report-sha256")
    p.add_argument(
        "--skip-block-reference",
        action="store_true",
        help="Only when reusing separately pinned reference evidence",
    )
    a = p.parse_args()
    if a.target_layer_graph and (
        a.draft_tokens != 2
        or a.recovery != "prefix"
        or a.mode == "pp"
        or a.projection_profile != "q8-residual"
    ):
        p.error("Target layer graph requires residual scalar MTP2 prefix mode")
    if a.recovery != "replay" and a.draft_tokens != 2:
        p.error("Prefix recovery requires explicit MTP2")
    if a.mode == "pp" and a.projection_profile != "fp16":
        p.error("Residual projection profile currently supports TP, not PP")
    if bool(a.prior_target_report) != bool(a.prior_target_report_sha256):
        p.error("Both prior target report and SHA256 are required together")
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    mtp_weights = QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact))
    mtp_weights.validate_target(weights)
    report = dict(
        scope=__doc__,
        mode=a.mode,
        projection_profile=a.projection_profile,
        mtp_draft_projection_profile="fp16",
        library_sha256=digest(a.library),
        mtp_artifact_sha256=digest(a.mtp_artifact),
        target_artifact_id=weights.artifact.artifact_id,
        mtp_artifact_id=mtp_weights.artifact.artifact_id,
        block_reference=[],
        recurrence_reference=[],
        draft_tokens=a.draft_tokens,
        target_recovery=a.recovery,
        target_layer_graph=a.target_layer_graph,
        max_projection_rows=128,
        prior_target_report_sha256=a.prior_target_report_sha256,
        cases=[],
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    kw = dict(
        weight_mode="resident", dtype=torch.float16, device="cuda:0", max_tokens=128
    )
    backend = CudaBackend(a.library)
    target = (
        NativeQwenPipelineModel.from_q8(
            weights, backend, pipeline_device="cuda:1", split_layer=32, **kw
        )
        if a.mode == "pp"
        else NativeQwenModel.from_q8(
            weights,
            backend,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            projection_profile=a.projection_profile,
            **kw,
        )
    )
    model = NativeQwenMTPModel.from_q8(target, mtp_weights)
    try:
        prior_target = None
        if a.prior_target_report:
            assert digest(a.prior_target_report) == a.prior_target_report_sha256
            prior_target = json.loads(Path(a.prior_target_report).read_text())
            assert (
                prior_target["completed"]
                and prior_target["artifact_id"] == weights.artifact.artifact_id
            )
            prior_target = prior_target["profiles"][a.projection_profile]["candidate"]
        if not a.skip_block_reference or a.draft_tokens == 2:
            source = Path(a.source)
            index = json.loads((source / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            raw = {}
            for name in index:
                if name.startswith("mtp."):
                    with safe_open(
                        str(source / index[name]), framework="pt", device="cpu"
                    ) as f:
                        raw[name] = f.get_tensor(name)
            reader_name = "model.language_model.embed_tokens.weight"
            session = model.create_session(32)
            cache = None
            try:
                for iteration, ids in enumerate(
                    () if a.skip_block_reference else ([1, 17, 42], [73])
                ):
                    h = (
                        torch.randn(
                            len(ids),
                            weights.config["hidden_size"],
                            generator=torch.Generator().manual_seed(531 + iteration),
                        )
                        * 0.31
                    ).half()
                    actual = session.draft_forward(ids, h)
                    with safe_open(
                        str(source / index[reader_name]), framework="pt", device="cpu"
                    ) as f:
                        sliced = f.get_slice(reader_name)
                        emb = torch.cat([sliced[i : i + 1] for i in ids])
                    start = perf_counter()
                    ref, _, cache = mtp_oracle(
                        raw,
                        weights.config,
                        ids,
                        h,
                        cache,
                        embedding=represented_source(emb),
                        quantizer=represented_source,
                        compute_logits=False,
                    )
                    state = session.draft_snapshot()
                    checks = {
                        "hidden": block_comparison(actual["hidden"], ref),
                        "keys": block_comparison(state["keys"], cache[0]),
                        "values": block_comparison(state["values"], cache[1]),
                    }
                    passed = all(
                        v["relative_l2"] <= 0.02 and v["centered_cosine"] >= 0.999
                        for v in checks.values()
                    )
                    report["block_reference"].append(
                        dict(
                            ids=ids,
                            position=state["position"],
                            checks=checks,
                            passed=passed,
                            reference_wall_seconds=perf_counter() - start,
                            native_top1=int(actual["logits"].argmax()),
                            shared_head_scope="Previously qualified native target Q8 head; no fresh independent CPU head/corpus gate",
                        )
                    )
                    save()
                    assert passed, (
                        "Independent trained MTP hidden/KV reference gate failed"
                    )
                if a.draft_tokens == 2:
                    session.reset()
                    cache = None
                    ids = [73]
                    h = (
                        torch.randn(
                            1,
                            weights.config["hidden_size"],
                            generator=torch.Generator().manual_seed(893),
                        )
                        * 0.31
                    ).half()
                    for step in range(3):
                        actual = session.draft_forward(ids, h)
                        with safe_open(
                            str(source / index[reader_name]),
                            framework="pt",
                            device="cpu",
                        ) as f:
                            sliced = f.get_slice(reader_name)
                            emb = torch.cat([sliced[i : i + 1] for i in ids])
                        ref, _, cache = mtp_oracle(
                            raw,
                            weights.config,
                            ids,
                            h,
                            cache,
                            embedding=represented_source(emb),
                            quantizer=represented_source,
                            compute_logits=False,
                        )
                        state = session.draft_snapshot()
                        checks = {
                            "hidden": block_comparison(actual["hidden"], ref),
                            "keys": block_comparison(state["keys"], cache[0]),
                            "values": block_comparison(state["values"], cache[1]),
                        }
                        passed = all(
                            v["relative_l2"] <= 0.02 and v["centered_cosine"] >= 0.999
                            for v in checks.values()
                        )
                        report["recurrence_reference"].append(
                            dict(
                                step=step,
                                ids=ids,
                                position=state["position"],
                                checks=checks,
                                passed=passed,
                                hidden_input="random normalized-state-shaped seed"
                                if step == 0
                                else "previous normalized native draft output; independently checked before reuse",
                            )
                        )
                        save()
                        assert passed, "Independent trained recursive MTP gate failed"
                        ids = [int(actual["logits"].argmax())]
                        h = actual["hidden"]
            finally:
                session.close()
            del raw, cache
        cases = json.loads(Path(a.cases).read_text())
        for case in cases:
            capacity = len(case["prompt_token_ids"]) + case["max_new_tokens"]
            ref, session = (
                target.create_session(capacity),
                model.create_session(capacity),
            )
            try:
                if a.target_layer_graph:
                    session.enable_target_layer_graph()
                expected = ref.generate(
                    case["prompt_token_ids"],
                    max_new_tokens=case["max_new_tokens"],
                    eos_token_ids=case.get("eos_token_ids", []),
                )

                if "expected_generated_ids" in case:
                    assert (
                        expected
                        == case["expected_generated_ids"][: case["max_new_tokens"]]
                    ), "Pinned target IDs changed"

                def forbidden(*args, **kwargs):
                    raise AssertionError(
                        "Python math called during native MTP generation"
                    )

                with ExitStack() as stack:
                    for name in (
                        "matmul",
                        "mm",
                        "empty",
                        "zeros",
                        "softmax",
                        "sigmoid",
                        "cat",
                        "tensor",
                    ):
                        stack.enter_context(patch.object(torch, name, forbidden))
                    actual = session.generate(
                        case["prompt_token_ids"],
                        max_new_tokens=case["max_new_tokens"],
                        eos_token_ids=case.get("eos_token_ids", []),
                        **(
                            {"draft_tokens": 2, "recovery": a.recovery}
                            if a.draft_tokens == 2
                            else {}
                        ),
                    )
                assert actual == expected, "Trained MTP/target output mismatch"
                same_state(session.snapshot(), ref.snapshot())
                if prior_target is not None:
                    from qualify_qwen_projection_variant import state_signature

                    pinned = next(
                        v for v in prior_target["cases"] if v["name"] == case["name"]
                    )
                    assert (
                        expected == pinned["generated_ids"]
                        and state_signature(ref.snapshot())
                        == pinned["committed_private_state"]
                    ), "Pinned target trajectory/state changed"
                draft_equal = None
                if a.draft_tokens == 2:
                    one = model.create_session(capacity)
                    try:
                        assert (
                            one.generate(
                                case["prompt_token_ids"],
                                max_new_tokens=case["max_new_tokens"],
                                eos_token_ids=case.get("eos_token_ids", []),
                            )
                            == expected
                        )
                        d0, d2 = one.draft_snapshot(), session.draft_snapshot()
                        assert d0["position"] == d2["position"]
                        assert all(
                            torch.equal(
                                d0[k].view(torch.uint8), d2[k].view(torch.uint8)
                            )
                            for k in ["keys", "values"]
                        ), "Corrected draft KV differs from MTP1"
                        draft_equal = True
                    finally:
                        one.close()
                stats = session.stats()
                assert stats["forced_proposals"] == 0
                record = dict(
                    case=case,
                    generated_ids=actual,
                    target_ids=expected,
                    output_ids_equal=True,
                    committed_target_state_bytes_equal=True,
                    committed_draft_state_equal_mtp1=draft_equal,
                    pinned_target_trajectory_state_reused=(prior_target is not None),
                    position=session.position,
                    draft_position=stats["draft_position"],
                    mtp_stats=stats,
                    generation_timings=session.generation_timings(),
                    native_without_python_math=True,
                )
                report["cases"].append(record)
                save()
                print(json.dumps(dict(case=case.get("name"), **stats)), flush=True)
            finally:
                ref.close()
                session.close()
        report["completed"] = True
        save()
    finally:
        model.close()
        target.close()


if __name__ == "__main__":
    main()
