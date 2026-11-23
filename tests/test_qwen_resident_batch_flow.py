"""Native A/B: exact state across resident dataflow, descriptor eager and graph."""

from contextlib import ExitStack, closing
import os

import pytest
import torch

from test_qwen_resident_embedding_io import (
    assert_exact,
    assert_state,
    new_model,
    write_fixture,
)
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend


@pytest.mark.cuda
def test_resident_batch_same_state_with_dynamic_graph_descriptors(tmp_path):
    libs = [
        os.environ.get(n)
        for n in ("VENTURI_FLOW_OPS_LIBRARY", "VENTURI_FLOW_QWEN_LIBRARY")
    ]
    if (
        not all(libs)
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        pytest.skip("Explicit SM70 paired assembly required")
    path = tmp_path / "q5.safetensors"
    write_fixture(path, 5)
    rt = QwenRuntime(libs[1], CudaBackend(libs[0]))
    with ExitStack() as stack:
        model = stack.enter_context(closing(new_model(path, rt)))
        modes = {
            (flow, engine): stack.enter_context(
                closing(
                    model.create_batch(
                        [64] * 4, resident_dataflow=flow, attention_execution=engine
                    )
                )
            )
            for flow in (False, True)
            for engine in ("legacy", "descriptor", "graph")
        }
        baseline = modes[False, "legacy"]

        def step(prompts, slots, sample_slots=None, decode_slots=()):
            results = [
                b.step(
                    prompts,
                    slots=slots,
                    sample_slots=sample_slots,
                    decode_slots=decode_slots,
                )
                for b in modes.values()
            ]
            assert all(r == results[0] for r in results)

        def states():
            for b in list(modes.values())[1:]:
                for slot in range(4):
                    assert_state(baseline.snapshot(slot), b.snapshot(slot))
                    assert (
                        baseline.info()["requests"][slot]["position"]
                        == b.info()["requests"][slot]["position"]
                    )

        step([[1, 5], [4], [8, 1, 7], [13, 7]], [0, 1, 2, 3], sample_slots=[])
        step([[3], [12, 6, 4]], [0, 2], sample_slots=[2, 0], decode_slots=[0])
        before = {k: b.resident_execution_info() for k, b in modes.items()}
        step([[9, 5, 7]], [3])
        step([[4], [13]], [2, 0], decode_slots=[2, 0])
        step([[4], [13]], [0, 3], decode_slots=[0, 3])
        step([[7], [12], [5]], [1, 2, 0], decode_slots=[1, 2, 0])
        step([[4], [13]], [2, 0], sample_slots=[], decode_slots=[2, 0])
        step([[13]], [3], decode_slots=[3])
        after = {k: b.resident_execution_info() for k, b in modes.items()}
        states()

        for flow in (False, True):
            g = after[flow, "graph"]
            assert g["graph_captures"] == 2  # M1, M2; only one full-attention layer
            assert g["graph_replays"] == 4  # M2 replay changes position and slot set
            assert g["graph_fallback_layers"] == 1  # M3 descriptor-eager
            assert g["descriptor_steps"] == 5
            assert (
                g["descriptor_bytes_uploaded"]
                == g["descriptor_steps"] * g["device_descriptor_bytes"]
            )
            assert g["capture_ns"] > 0
        clean = after[True, "legacy"]
        raw = after[False, "legacy"]
        assert clean["direct_v"] - before[True, "legacy"]["direct_v"] == 2
        assert clean["v_copies"] < raw["v_copies"]
        assert clean["hidden_copies"] < raw["hidden_copies"]
        assert clean["completion_fences"] < raw["completion_fences"]

        # Subsequent eager logits see every bit of modified recurrent and KV state.
        logits = [b.forward([[12], [7]], slots=[2, 1]) for b in modes.values()]
        for y in logits[1:]:
            assert_exact(logits[0], y)
        for b in modes.values():
            b.reset()
        step([[2], [9, 5]], [0, 2])
        step([[14], [15]], [0, 2], decode_slots=[0, 2])
        states()
        assert modes[True, "graph"].resident_execution_info()["graph_captures"] == 2

        caught = modes[True, "graph"]
        pos = caught.info()["requests"][0]["position"]
        caught._inject_failure(slot=0, layer=1, position=pos)
        with pytest.raises(RuntimeError, match="injected"):
            caught.step([[3], [4]], slots=[0, 2], decode_slots=[0, 2])
        assert caught.info()["requests"][0]["failed"]
        assert caught.info()["requests"][0]["position"] == pos
        with pytest.raises(RuntimeError, match="reset required"):
            caught.step([[3]], slots=[0], decode_slots=[0])
        for b in modes.values():
            b.reset()
        step([[2], [5]], [0, 1])
        step([[14], [15]], [0, 1], decode_slots=[0, 1])
        states()
        assert caught.resident_execution_info()["graph_captures"] == 2
