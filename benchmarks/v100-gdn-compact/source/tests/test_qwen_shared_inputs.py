"""Sharing is byte-exact to a pinned unshared binary within each compute profile."""

import os

import pytest
import torch

from test_qwen_model import hybrid_fixture
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenModel
from venturi.ops import CudaBackend


@pytest.mark.parametrize("profile", ["fp16", "q8-residual"])
@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("count", [1, 8])
@pytest.mark.parametrize("tied", [False, True])
def test_shared_inputs_exact_to_unshared_chunks_heads_resets_and_private_batch(
    tmp_path, profile, tp, count, tied
):
    old = os.environ.get("VENTURI_SHARED_BASELINE_LIBRARY")
    new = os.environ.get("VENTURI_SHARED_LIBRARY")
    if not old or not new:
        pytest.skip("explicit pinned unshared/shared libraries required")
    weights, _ = hybrid_fixture(tmp_path, tied=tied)
    models = [
        NativeQwenModel.from_q8(
            weights,
            CudaBackend(library),
            weight_mode="resident",
            dtype=torch.float16,
            max_tokens=8,
            reserve_bytes=0,
            projection_profile=profile,
            **(
                dict(tensor_parallel_device="cuda:1", tensor_parallel_transport="host")
                if tp
                else {}
            ),
        )
        for library in (old, new)
    ]
    sessions = [[m.create_session(64) for _ in range(2)] for m in models]
    batches = [m.create_batch([64] * count) for m in models]
    try:
        # Different owners overwrite identical workspace addresses. Eleven rows
        # cross a chunk boundary; head and layer normalizations share scratch.
        for slot, ids in [
            (0, list(range(1, 12))),
            (1, [7, 3]),
            (0, [2]),
            (1, [4, 9, 1]),
        ]:
            traces = [[], []]
            results = [
                owners[slot].forward(ids, all_logits=True, trace=trace)
                for owners, trace in zip(sessions, traces)
            ]
            assert torch.equal(
                results[0].view(torch.uint8), results[1].view(torch.uint8)
            )
            assert all(
                torch.equal(
                    a["hidden"].view(torch.uint8), b["hidden"].view(torch.uint8)
                )
                for a, b in zip(*traces)
            )
            same_state(sessions[0][slot].snapshot(), sessions[1][slot].snapshot())
        for owners in sessions:
            owners[0].reset()
        generated = [
            owners[0].generate([3, 8, 1], max_new_tokens=8) for owners in sessions
        ]
        assert generated[0] == generated[1]
        same_state(sessions[0][0].snapshot(), sessions[1][0].snapshot())
        for slots in (list(range(count)), list(dict.fromkeys([count - 1, 0]))):
            prompts = [
                [(i + j + 1) % weights.config["vocab_size"] for j in range(1 + i % 5)]
                for i in slots
            ]
            outputs = [b.forward(prompts, slots=slots) for b in batches]
            assert torch.equal(
                outputs[0].view(torch.uint8), outputs[1].view(torch.uint8)
            )
            for i in range(count):
                same_state(batches[0].snapshot(i), batches[1].snapshot(i))
        for b in batches:
            b.reset(slots=[count - 1])
        outputs = [b.forward([[1, 7, 2]], slots=[count - 1]) for b in batches]
        assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
        for i in range(count):
            same_state(batches[0].snapshot(i), batches[1].snapshot(i))
        if tp:
            old_info, new_info = [owners[0].info() for owners in sessions]
            assert (
                0
                < new_info["peer_input_copy_bytes"]
                < old_info["peer_input_copy_bytes"]
            )
            assert (
                new_info["peer_output_copy_bytes"] == old_info["peer_output_copy_bytes"]
            )
    finally:
        for b in batches:
            b.close()
        for owners in sessions:
            for s in owners:
                s.close()
        for m in models:
            m.close()
