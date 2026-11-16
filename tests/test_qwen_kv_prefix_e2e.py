"""End-to-end unified-KV prefix sharing: the shared path matches a full
prefill bit-for-bit, reuses physical pages, and survives its donor.

Scheduling split: the unified KV layer (ops/csrc/kv) owns the record/snapshot/
read-only-mapping orchestration; the model layer only decides which sessions,
entries and recurrent buffers participate.
"""

import os

import pytest
import torch

from venturi.models.qwen.native_runtime import QwenRuntime
from test_native_head_gdn import eligible_mtp
from venturi.models import NativeQwenModel
from venturi.ops import CudaBackend


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = os.environ.get("VENTURI_KV_TIER_LIBRARY")
    if not path:
        pytest.skip("Explicit SM70 library and a V100 required")
    weights, _, _, _ = eligible_mtp(tmp_path_factory.mktemp("prefix-source") / "source")
    assert "linear_attention" in weights.config["layer_types"]
    assert "full_attention" in weights.config["layer_types"]
    model = NativeQwenModel.from_q8(
        weights,
        QwenRuntime(path, CudaBackend(path)),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        reserve_bytes=0,
        kv_cache="vram-first",
        kv_ram_budget_bytes=256 * 2**20,
    )
    yield model
    model.close()


PREFIX = [3, 1, 4, 1, 5]


def test_shared_prefix_matches_full_prefill_and_survives_donor(engine):
    full = engine.create_session(64)
    donor = engine.create_session(64)
    try:
        full.forward(PREFIX, all_logits=True)
        donor.forward(PREFIX, all_logits=True)
        donor.publish_prefix(7, len(PREFIX))
        pool_before = engine.kv_cache_info()["gpu_pool_bytes"]

        borrower = engine.create_session(64, prefix_key=7, prefix_tokens=len(PREFIX))
        try:
            # Borrowing reuses the donor's physical pages: no new pool growth.
            assert engine.kv_cache_info()["gpu_pool_bytes"] == pool_before
            exact(
                borrower.forward([9, 2, 6], all_logits=True),
                full.forward([9, 2, 6], all_logits=True),
            )
            # Destroying the donor keeps the shared pages and data alive.
            donor.close()
            donor = None
            exact(
                borrower.forward([5, 8], all_logits=True),
                full.forward([5, 8], all_logits=True),
            )
        finally:
            borrower.close()
    finally:
        if donor is not None:
            donor.close()
        full.close()


def test_prefix_generation_matches_full_prefill(engine):
    full = engine.create_session(64)
    donor = engine.create_session(64)
    borrower = None
    try:
        full.forward(PREFIX, all_logits=True)
        donor.forward(PREFIX, all_logits=True)
        donor.publish_prefix(11, len(PREFIX))
        borrower = engine.create_session(64, prefix_key=11, prefix_tokens=len(PREFIX))
        donor.close()
        donor = None
        reference = full.generate([9, 9], max_new_tokens=8)
        shared = borrower.generate([9, 9], max_new_tokens=8)
        assert shared == reference
    finally:
        if donor is not None:
            donor.close()
        if borrower is not None:
            borrower.close()
        full.close()


def test_prefix_fails_closed(engine):
    session = engine.create_session(64)
    try:
        session.forward(PREFIX, all_logits=True)
        with pytest.raises(RuntimeError):
            engine.create_session(64, prefix_key=404, prefix_tokens=len(PREFIX))
        with pytest.raises(RuntimeError):
            session.publish_prefix(7, len(PREFIX) + 100)
    finally:
        session.close()


def test_prefix_requires_vram_first_kv(tmp_path):
    if not os.environ.get("VENTURI_KV_TIER_LIBRARY"):
        pytest.skip("Explicit SM70 library and a V100 required")
    weights, _, _, _ = eligible_mtp(tmp_path / "source")
    model = NativeQwenModel.from_q8(
        weights,
        QwenRuntime(
            os.environ["VENTURI_KV_TIER_LIBRARY"],
            CudaBackend(os.environ["VENTURI_KV_TIER_LIBRARY"]),
        ),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        reserve_bytes=0,
    )
    try:
        session = model.create_session(64)
        try:
            session.forward(PREFIX, all_logits=True)
            with pytest.raises(RuntimeError):
                session.publish_prefix(7, len(PREFIX))
        finally:
            session.close()
    finally:
        model.close()
