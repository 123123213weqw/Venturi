"""Opt-in two-device replica routing against independent native greedy sessions."""
import os

import pytest
import torch

from qwen_native_runtime import native_runtime
from test_qwen_model import hybrid_fixture
from venturi.models import NativeQwenModel
from venturi.ops import CudaBackend
from venturi.runtime import ContinuousReplica, ContinuousReplicaPool, ContinuousScheduler


@pytest.mark.cuda
@pytest.mark.parametrize("requests", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_two_native_replicas_preserve_distinct_greedy_trajectories(tmp_path, requests, dtype):
    if os.environ.get("VENTURI_REPLICA_TEST") != "1":
        pytest.skip("explicit two-device replica test opt-in required")
    if torch.cuda.device_count() < 2:
        pytest.skip("two separately admitted idle GPUs required")
    runtime = native_runtime(CudaBackend(os.environ["VENTURI_OPS_LIBRARY"]))
    weights, _ = hybrid_fixture(tmp_path)
    models, batches, pool = [], [], None
    try:
        for device in ("cuda:0", "cuda:1"):
            models.append(NativeQwenModel.from_q8(weights, runtime,
                weight_mode="resident", dtype=dtype, device=device, max_tokens=8,
                reserve_bytes=0, metadata_mode="sealed"))
            batches.append(models[-1].create_batch([64]*4))
        cases = [([(i+j+1) % 8 for j in range(1+i % 7)], 2+i % 5) for i in range(requests)]
        expected = []
        for ids, count in cases:
            ref = models[0].create_session(64)
            try:
                expected.append(tuple(ref.generate(ids, max_new_tokens=count)))
            finally:
                ref.close()
        pool = ContinuousReplicaPool(tuple(ContinuousReplica(str(i), "same-fixture",
            ContinuousScheduler(b, token_budget=8, prefill_chunk_size=3))
            for i,b in enumerate(batches)), max_requests=64)
        with pool:
            tickets = [pool.submit(ids, max_new_tokens=count) for ids,count in cases]
            actual = [t.future.result(timeout=120).output_ids for t in tickets]
            assert actual == expected
            assert len({t.request_id for t in tickets}) == requests
            assert {t.replica for t in tickets} == {"0", "1"}
            assert pool.info()["completed"] == requests
            assert pool.info()["unfinished"] == 0
        assert all(not r["worker"]["alive"] for r in pool.info()["replicas"])
    finally:
        # Never release borrowed native buffers while an owner remains live.
        if pool is not None:
            pool.close(timeout=120)
        for batch in batches:
            batch.close()
        for model in models:
            model.close()
