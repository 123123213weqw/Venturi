"""Opt-in real CUDA PD state transfer and independent-device continuation.

Generated tiny hybrid checkpoint only; not a trained 27B performance/quality test.
"""
from dataclasses import replace
import os

import pytest
import torch

from test_qwen_model import hybrid_fixture
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenModel
from venturi.models.qwen_pd import NativeQwenPDWorker, NativeQwenStateTransport
from venturi.ops import CudaBackend
from venturi.runtime import PDScheduler, PDWorkerPool


@pytest.fixture(params=[torch.float16, torch.bfloat16])
def pd_engine(request):
    if os.environ.get("VENTURI_PD_TEST") != "1":
        pytest.skip("explicit PD GPU test opt-in required")
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY") or os.environ.get("VENTURI_SM70_LIBRARY")
    if not library or torch.cuda.device_count() < 2:
        pytest.skip("explicit native library and two GPUs required")
    backend = CudaBackend(library)
    if backend.architecture == 70 and request.param == torch.bfloat16:
        pytest.skip("SM70 has no qualified BF16 path")
    return backend, request.param


def workers(tmp_path, pd_engine, p=1, d=1, *, capacities=(64, 64)):
    backend, dtype = pd_engine
    if torch.cuda.device_count() < p + d:
        pytest.skip(f"{p + d} physical GPUs required")
    weights, _ = hybrid_fixture(tmp_path)
    models, batches, targets = [], [], []
    try:
        for device in range(p + d):
            model = NativeQwenModel.from_q8(weights, backend, weight_mode="resident", dtype=dtype,
                        device=f"cuda:{device}", max_tokens=8 if device < p else 4,
                        reserve_bytes=0, metadata_mode="sealed")
            models.append(model)
            batch = model.create_batch(list(capacities))
            batches.append(batch)
            targets.append(NativeQwenPDWorker(batch))
    except BaseException:
        cleanup(models, batches)
        raise
    return models, batches, targets[:p], targets[p:]


def cleanup(models, batches):
    for item in list(reversed(batches)) + list(reversed(models)):
        item.close()


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["host", "peer"])
def test_pd_handoff_all_state_planes_exact_logits_and_source_immutable(tmp_path, pd_engine, mode):
    models, batches, p, d = workers(tmp_path, pd_engine)
    reference = models[0].create_session(64)
    transport = NativeQwenStateTransport(mode=mode)
    try:
        prompt = [1, 2, 3]
        first = p[0].step([prompt], slots=[0], sample_slots=[0])[0]
        expected = reference.forward(prompt)
        assert first == int(expected.argmax())
        original = batches[0].snapshot(0)
        inactive = batches[1].snapshot(1)
        state = p[0].export_state(0, position=len(prompt))
        assert any("recurrent" in c.name and c.dtype == "float32" for c in state.components)
        assert any("key" in c.name for c in state.components)
        receipt = transport.transfer(state, d[0], slot=0)
        assert receipt.position == len(prompt) and receipt.payload_bytes == state.nbytes > 0
        assert bool(receipt.staging_bytes) == (mode == "host")
        same_state(batches[1].snapshot(0), original)
        same_state(batches[1].snapshot(1), inactive)
        for token in (first, 7, 2, 5):
            expected = reference.forward([token])
            actual = batches[1].forward([[token]], slots=[0])[0]
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            same_state(batches[1].snapshot(0), reference.snapshot())
            same_state(batches[0].snapshot(0), original)
    finally:
        reference.close()
        cleanup(models, batches)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["host", "peer"])
def test_pd_stale_lease_identity_and_occupied_target_rejected_before_copy(tmp_path, pd_engine, mode):
    models, batches, p, d = workers(tmp_path, pd_engine)
    transport = NativeQwenStateTransport(mode=mode)
    try:
        p[0].step([[1, 2]], slots=[0], sample_slots=[0])
        state = p[0].export_state(0, position=2)
        bad = replace(state, identity=replace(state.identity, checkpoint="not-the-source"))
        with pytest.raises(ValueError, match="identity mismatch"):
            transport.transfer(bad, d[0], slot=0)
        bad = replace(state, components=state.components[:-1])
        with pytest.raises(ValueError, match="altered"):
            transport.transfer(bad, d[0], slot=0)
        p[0].step([[3]], slots=[0], sample_slots=[0])
        with pytest.raises(ValueError, match="stale"):
            transport.transfer(state, d[0], slot=0)
        state = p[0].export_state(0, position=3)
        transport.transfer(state, d[0], slot=0)
        before = batches[1].snapshot(0)
        with pytest.raises(ValueError, match="fresh"):
            transport.transfer(state, d[0], slot=0)
        same_state(batches[1].snapshot(0), before)
        # Repeat a clean import after explicit reuse; no old tenant state survives.
        d[0].reset(slots=[0])
        transport.transfer(state, d[0], slot=0)
        same_state(batches[1].snapshot(0), before)
        p[0].reset(slots=[0])
        with pytest.raises(ValueError, match="stale"):
            transport.transfer(state, d[0], slot=1)
    finally:
        cleanup(models, batches)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["host", "peer"])
@pytest.mark.parametrize("pcount,dcount", [(1, 1), (2, 2), (4, 4)])
def test_pd_pool_independent_devices_greedy_refill_and_no_python_tensor_math(
        tmp_path, pd_engine, mode, pcount, dcount, monkeypatch):
    models, batches, p, d = workers(tmp_path, pd_engine, pcount, dcount)
    scheduler = PDScheduler(p, d, NativeQwenStateTransport(mode=mode), prefill_chunk_size=3)
    pool = PDWorkerPool(scheduler)
    refs = [models[0].create_session(64) for _ in range(8)]
    cases = [([1 + i, 2, 3, 4, 5][:1 + i % 5], 2 + i % 5) for i in range(8)]
    try:
        expected = [s.generate(ids, max_new_tokens=n, eos_token_ids=[])
                    for s, (ids, n) in zip(refs, cases)]

        def forbidden(*args, **kwargs):
            raise AssertionError("PD used Python tensor allocation or math")
        with monkeypatch.context() as patch:
            for name in ("empty", "zeros", "matmul", "mm", "bmm", "argmax"):
                patch.setattr(torch, name, forbidden)
            for name in ("to", "cpu", "float", "matmul", "argmax"):
                patch.setattr(torch.Tensor, name, forbidden)
            pool.start()
            tickets = [pool.submit(ids, max_new_tokens=n, eos_token_ids=[]) for ids, n in cases]
            actual = [ticket.future.result(timeout=90).output_ids for ticket in tickets]
        assert actual == [tuple(row) for row in expected]
        assert scheduler.info()["handoffs"] == len(cases)
        assert scheduler.info()["completed_requests"] == len(cases)
        assert scheduler.info()["failed"] is False
    finally:
        pool.close(timeout=90)
        for ref in refs:
            ref.close()
        cleanup(models, batches)
