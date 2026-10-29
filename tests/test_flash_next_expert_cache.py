"""Decoded floating expert cache ownership/LRU/errors/concurrent misses."""

from concurrent.futures import ThreadPoolExecutor
import time

import pytest
import torch

from venturi.models.flash_next_expert_cache import CachedExpertStore
from venturi.models.flash_next_moe import MoEGeometry, TensorExpertStore


def source():
    g = MoEGeometry(8, 3, 1, 6, 6)
    bank = TensorExpertStore(g, {e: (torch.full((12, 8), float(e+1)), torch.full((8, 6), float(e+2))) for e in range(3)})
    original = bank.read
    bank.calls = []
    def read(e, **kwargs):
        bank.calls.append(e)
        return original(e, **kwargs)
    bank.read = read
    return bank


def test_lru_exact_bytes_owned_storage_and_live_eviction_lease():
    bank = source()
    entry_bytes = 3*8*6*4
    cache = CachedExpertStore(bank, max_resident_bytes=2*entry_bytes)
    a = cache.read(0, device='cpu', dtype=torch.float32)
    cache.read(1, device='cpu', dtype=torch.float32)
    hit = cache.read(0, device='cpu', dtype=torch.float32)
    assert hit[0].data_ptr() == a[0].data_ptr() and a[0].data_ptr() != bank.experts[0][0].data_ptr()
    bank.experts[0][0].zero_()
    assert a[0].eq(1).all()
    cache.read(2, device='cpu', dtype=torch.float32)
    cache.read(1, device='cpu', dtype=torch.float32)
    assert cache.stats == dict(hits=1, misses=4, evictions=2, bypasses=0, entries=2,
        retained_bytes=2*entry_bytes, max_resident_bytes=2*entry_bytes)
    cache.clear()
    assert cache.stats['retained_bytes'] == 0 and a[0].eq(1).all()


def test_dtype_keys_and_explicit_oversize_bypass():
    bank = source()
    cache = CachedExpertStore(bank, max_resident_bytes=3*8*6*4)
    cache.read(0, device='cpu', dtype=torch.float32)
    a = cache.read(0, device='cpu', dtype=torch.bfloat16)
    b = cache.read(0, device='cpu', dtype=torch.float16)
    assert a[0].dtype == torch.bfloat16 and b[0].dtype == torch.float16 and cache.stats['entries'] == 2
    bypass = CachedExpertStore(bank, max_resident_bytes=0)
    bypass.read(0, device='cpu', dtype=torch.float32)
    bypass.read(0, device='cpu', dtype=torch.float32)
    assert bypass.stats['bypasses'] == 2 and bypass.stats['entries'] == 0


def test_concurrent_same_key_fill_is_coalesced():
    bank = source()
    original = bank.read
    def slow(e, **kwargs):
        time.sleep(.01)
        return original(e, **kwargs)
    bank.read = slow
    cache = CachedExpertStore(bank, max_resident_bytes=4096)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: cache.read(1, device='cpu', dtype=torch.float32), range(8)))
    assert bank.calls == [1] and cache.stats['misses'] == 1 and cache.stats['hits'] == 7
    assert all(r[0].data_ptr() == results[0][0].data_ptr() for r in results)


def test_failed_fill_or_invalid_source_never_publishes_partial_entry(monkeypatch):
    bank = source()
    cache = CachedExpertStore(bank, max_resident_bytes=4096)
    def fail(*args, **kwargs):
        raise RuntimeError('source unavailable')
    original = bank.read
    monkeypatch.setattr(bank, 'read', fail)
    with pytest.raises(RuntimeError, match='unavailable'):
        cache.read(0, device='cpu', dtype=torch.float32)
    assert cache.stats['entries'] == 0 and cache.stats['retained_bytes'] == 0
    monkeypatch.setattr(bank, 'read', lambda *args, **kwargs: (torch.zeros(1), torch.zeros(1)))
    with pytest.raises(ValueError, match='incompatible'):
        cache.read(0, device='cpu', dtype=torch.float32)
    assert cache.stats['entries'] == 0
    monkeypatch.setattr(bank, 'read', original)
    cache.read(0, device='cpu', dtype=torch.float32)
    assert cache.stats['entries'] == 1


def test_admission_rejects_invalid_expert_backend_and_budget():
    bank = source()
    for budget in (-1, True):
        with pytest.raises(ValueError):
            CachedExpertStore(bank, max_resident_bytes=budget)
    cache = CachedExpertStore(bank, max_resident_bytes=4096)
    for expert, device in [(True, 'cpu'), (-1, 'cpu'), (3, 'cpu'), (0, 'mps')]:
        with pytest.raises(ValueError):
            cache.read(expert, device=device, dtype=torch.float32)
    assert not bank.calls
