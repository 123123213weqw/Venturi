"""Real-checkpoint scalar/batched output and private-cache byte parity.

Diagnostic only: snapshots and scalar controls are not throughput measurements.
Reuses the independently qualified target operators without changing arithmetic.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_qwen_tensor_parallel import same_state  # noqa: E402
from venturi.models import NativeQwenModel, QwenQ8Weights  # noqa: E402
from venturi.ninfer import NInferArtifact  # noqa: E402
from venturi.ops import CudaBackend  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--output-directory", required=True)
    args = p.parse_args()
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    cases = json.loads(Path(args.cases).read_text())
    # Two differently ordered copies, not eight views of one recurrent cache.
    order = [0, 1, 2, 3, 3, 2, 1, 0]
    report = dict(
        scope=__doc__,
        artifact_id=weights.artifact.artifact_id,
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        precision="Original-source Q8/group32 FP16 scales; FP16 activations/KV, FP32 GDN state; original small-parameter profile unchanged",
        independent_reference_scope="Reuses frozen v100-native-mtp/native-target represented-weight reference gates. This is new exact scalar/batch state and output qualification, not a new BF16/corpus gate.",
        requests=[],
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    model = NativeQwenModel.from_q8(
        weights,
        CudaBackend(args.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
    )
    batch, scalar = None, []
    try:
        batch = model.create_batch([256] * 8)
        scalar = [model.create_session(256) for _ in range(8)]
        prompts = [cases[i]["prompt_token_ids"] for i in order]
        maximum = [cases[i]["max_new_tokens"] for i in order]
        # Truncate one request to exercise a different non-EOS budget too.
        maximum[4] = 13
        eos = [cases[i].get("eos_token_ids", []) for i in order]
        expected = []
        for j, s in enumerate(scalar):
            expected.append(
                s.generate(prompts[j], max_new_tokens=maximum[j], eos_token_ids=eos[j])
            )
            pinned = cases[order[j]].get("expected_generated_ids")
            if pinned is not None:
                assert expected[-1] == pinned[: maximum[j]], (
                    "Pinned scalar output changed"
                )

        def forbidden(*args, **kwargs):
            raise AssertionError("Torch math called by native batched generation")

        with ExitStack() as stack:
            for name in (
                "matmul",
                "mm",
                "bmm",
                "empty",
                "zeros",
                "softmax",
                "sigmoid",
                "cat",
                "tensor",
            ):
                stack.enter_context(patch.object(torch, name, forbidden))
            actual = batch.generate(prompts, max_new_tokens=maximum, eos_token_ids=eos)
        assert actual == expected, "Real-checkpoint grouped target IDs differ"
        info = batch.info()
        for j, s in enumerate(scalar):
            same_state(batch.snapshot(j), s.snapshot())
            assert (
                info["requests"][j]["position"] == len(prompts[j]) + len(actual[j]) - 1
            )
            report["requests"].append(
                dict(
                    slot=j,
                    case=cases[order[j]]["name"],
                    prompt_ids=prompts[j],
                    maximum=maximum[j],
                    eos=eos[j],
                    generated_ids=actual[j],
                    scalar_ids=expected[j],
                    state_bytes_exact=True,
                    info=info["requests"][j],
                )
            )
            save()
        report["batch_info"] = info
        # Different positions and a non-monotonic subset must leave other slots intact.
        selected = [7, 1, 4]
        continuation = [[17], [42, 73], [19, 11, 3]]
        reference = torch.stack(
            [scalar[i].forward(ids) for i, ids in zip(selected, continuation)]
        )
        logits = batch.forward(continuation, slots=selected)
        assert torch.equal(logits.view(torch.uint8), reference.view(torch.uint8)), (
            "Continuation logits differ"
        )
        for j, s in enumerate(scalar):
            same_state(batch.snapshot(j), s.snapshot())
        report["reordered_unequal_position_continuation"] = dict(
            slots=selected,
            prompts=continuation,
            logits_bytes_exact=True,
            all_slot_states_bytes_exact=True,
        )
        report["completed"] = True
        save()
    finally:
        if batch is not None:
            batch.close()
        for s in scalar:
            s.close()
        model.close()


if __name__ == "__main__":
    main()
