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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--mode", choices=("tp", "pp"), required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument(
        "--skip-block-reference",
        action="store_true",
        help="Only when reusing separately pinned reference evidence",
    )
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    mtp_weights = QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact))
    mtp_weights.validate_target(weights)
    report = dict(
        scope=__doc__,
        mode=a.mode,
        library_sha256=digest(a.library),
        mtp_artifact_sha256=digest(a.mtp_artifact),
        target_artifact_id=weights.artifact.artifact_id,
        mtp_artifact_id=mtp_weights.artifact.artifact_id,
        block_reference=[],
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
            **kw,
        )
    )
    model = NativeQwenMTPModel.from_q8(target, mtp_weights)
    try:
        if not a.skip_block_reference:
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
                for iteration, ids in enumerate(([1, 17, 42], [73])):
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
                        "hidden": metrics(actual["hidden"], ref),
                        "keys": metrics(state["keys"], cache[0]),
                        "values": metrics(state["values"], cache[1]),
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
                expected = ref.generate(
                    case["prompt_token_ids"],
                    max_new_tokens=case["max_new_tokens"],
                    eos_token_ids=case.get("eos_token_ids", []),
                )

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
                    )
                assert actual == expected, "Trained MTP/target output mismatch"
                same_state(session.snapshot(), ref.snapshot())
                stats = session.stats()
                assert stats["forced_proposals"] == 0
                record = dict(
                    case=case,
                    generated_ids=actual,
                    target_ids=expected,
                    output_ids_equal=True,
                    committed_target_state_bytes_equal=True,
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
