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
    def bits(a,b):
        if isinstance(a,torch.Tensor):
            if a.is_floating_point():assert bool(torch.isfinite(a).all())
            assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                b.detach().contiguous().reshape(-1).view(torch.uint8))
        elif isinstance(a,dict):
            for key in a:bits(a[key],b[key])
        elif isinstance(a,(tuple,list)):
            for x,y in zip(a,b):bits(x,y)
    bits(actual,expected)


def test_exact_checker_rejects_signed_zero_and_nonfinite_values():
    with pytest.raises(AssertionError):exact(torch.tensor([-0.]),torch.tensor([0.]))
    with pytest.raises(AssertionError):exact(torch.tensor([float('inf')]),torch.tensor([float('inf')]))


@pytest.fixture(scope="module")
def engine_factory(tmp_path_factory):
    path = os.environ.get("VENTURI_KV_TIER_LIBRARY")
    if not path:
        pytest.skip("Explicit SM70 library and a V100 required")
    weights, _, _, _ = eligible_mtp(tmp_path_factory.mktemp("prefix-source") / "source")
    assert "linear_attention" in weights.config["layer_types"]
    assert "full_attention" in weights.config["layer_types"]
    runtime=QwenRuntime(os.environ.get('VENTURI_QWEN_LIBRARY',path),
        CudaBackend(os.environ.get('VENTURI_OPS_LIBRARY',path)))
    models=[]
    def load(**options):
        model=NativeQwenModel.from_q8(weights,runtime,**(dict(weight_mode='resident',
            dtype=torch.float16,device='cuda:0',max_tokens=32,reserve_bytes=0,
            kv_cache='vram-first',kv_ram_budget_bytes=256*2**20)|options))
        models.append(model);return model
    yield load
    for model in reversed(models):model.close()


@pytest.fixture(scope="module")
def engine(engine_factory):
    return engine_factory()


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
        QwenRuntime(os.environ.get('VENTURI_QWEN_LIBRARY',os.environ['VENTURI_KV_TIER_LIBRARY']),
            CudaBackend(os.environ.get('VENTURI_OPS_LIBRARY',os.environ['VENTURI_KV_TIER_LIBRARY']))),
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


def test_publication_requires_healthy_exact_committed_boundary(engine):
    session=engine.create_session(64)
    try:
        session.forward(PREFIX,all_logits=True)
        before=session.snapshot()
        for tokens in(len(PREFIX)-1,len(PREFIX)+1):
            with pytest.raises(RuntimeError,match='committed donor position'):
                session.publish_prefix(101,tokens)
        exact(session.snapshot(),before)
        session._inject_failure(layer=0,position=len(PREFIX))
        with pytest.raises(RuntimeError,match='injected'):session.forward([9])
        assert session.failed
        with pytest.raises(RuntimeError,match='failed session cannot publish'):
            session.publish_prefix(102,len(PREFIX))
        session.reset();session.forward(PREFIX)
        session.publish_prefix(102,len(PREFIX))
    finally:session.close()


def test_attach_length_mismatch_cannot_map_or_poison_pool(engine):
    donor=engine.create_session(64)
    try:
        donor.forward(PREFIX);donor.publish_prefix(201,len(PREFIX))
        before=engine.kv_cache_info()['entries']
        for tokens in(len(PREFIX)-1,len(PREFIX)+1):
            with pytest.raises(RuntimeError,match='length differs'):
                engine.create_session(64,prefix_key=201,prefix_tokens=tokens)
            assert engine.kv_cache_info()['entries']==before
        borrower=engine.create_session(64,prefix_key=201,prefix_tokens=len(PREFIX))
        try:exact(borrower.forward([8],all_logits=True),donor.forward([8],all_logits=True))
        finally:borrower.close()
    finally:donor.close()


def test_reset_donor_cannot_rewrite_existing_borrower_or_revive_old_key(engine):
    reference,donor=engine.create_session(64),engine.create_session(64);borrower=None
    try:
        reference.forward(PREFIX);donor.forward(PREFIX);donor.publish_prefix(301,len(PREFIX))
        borrower=engine.create_session(64,prefix_key=301,prefix_tokens=len(PREFIX))
        donor.reset();donor.forward([8,7,6,5,4])
        with pytest.raises(RuntimeError,match='no usable prefix record'):
            engine.create_session(64,prefix_key=301,prefix_tokens=len(PREFIX))
        exact(borrower.forward([9,2,6],all_logits=True),reference.forward([9,2,6],all_logits=True))
        exact(borrower.snapshot(),reference.snapshot())
    finally:
        if borrower is not None:borrower.close()
        donor.close();reference.close()


def test_donor_append_and_republication_preserve_old_borrowers(engine):
    reference,donor=engine.create_session(64),engine.create_session(64);borrower=None;new=None
    try:
        reference.forward(PREFIX);donor.forward(PREFIX);donor.publish_prefix(401,len(PREFIX))
        borrower=engine.create_session(64,prefix_key=401,prefix_tokens=len(PREFIX))
        donor.forward([9,8]);donor.publish_prefix(401,len(PREFIX)+2)
        new=engine.create_session(64,prefix_key=401,prefix_tokens=len(PREFIX)+2)
        exact(new.forward([7],all_logits=True),donor.forward([7],all_logits=True))
        exact(borrower.forward([2,4],all_logits=True),reference.forward([2,4],all_logits=True))
    finally:
        if new is not None:new.close()
        if borrower is not None:borrower.close()
        donor.close();reference.close()


def test_native_prefix_pressure_is_exact_without_COW_headroom(engine_factory):
    reference=engine_factory();g=reference.kv_cache_info()['page_bytes']
    pressure=engine_factory(kv_gpu_budget_bytes=2*g*reference._cfg['layer_types'].count('full_attention'))
    a,d=reference.create_session(64),pressure.create_session(64);b=None
    try:
        a.forward(PREFIX);d.forward(PREFIX);d.publish_prefix(501,len(PREFIX))
        b=pressure.create_session(64,prefix_key=501,prefix_tokens=len(PREFIX))
        exact(b.forward([9,2],all_logits=True),a.forward([9,2],all_logits=True))
        exact(b.snapshot(),a.snapshot())
        assert pressure.kv_cache_info()['evictions']>0
        # Reload and append the original donor as well: stale RO flags must
        # never reappear after its private spill/reload cycle.
        d.forward([7]);assert not d.failed
    finally:
        if b is not None:b.close()
        d.close();a.close();pressure.close();reference.close()
