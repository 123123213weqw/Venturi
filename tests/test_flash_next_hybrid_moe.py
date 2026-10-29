"""Synthetic CPU scheduling/rounding contracts, not CPU/GPU overlap or model TPS."""
import threading

import pytest
import torch
from torch.nn import functional as F

from venturi.models.flash_next_hybrid_moe import (
    ColdExpertPool, FixedHotBank, HybridExpertDispatch, RouteCounter, cpu_expert, bind_fixed_expert_policy,
    CPUDecodedExpertCache,
)
from venturi.models.flash_next_moe import MoEGeometry, TensorExpertStore, routed_experts


@pytest.fixture
def setup():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(921)
    g = MoEGeometry(8, 4, 2, 6, 5)
    source = TensorExpertStore(g, {e:(torch.randn(12,8).half()*.1,torch.randn(8,6).half()*.1) for e in range(4)})
    pool = ColdExpertPool(2)
    yield source, pool
    pool.close()
    torch.set_num_threads(before)


@pytest.mark.parametrize('hot_ids', [(), (0,2), (0,1,2,3)])
@pytest.mark.parametrize('rows', [0,1,5])
def test_dispatch_all_cold_mixed_hot_and_multitoken_order(setup, hot_ids, rows):
    source,pool = setup
    bank = FixedHotBank(source, hot_ids, device='cpu', dtype=torch.float16, max_bytes=4*288)
    dispatch = HybridExpertDispatch(bank, source, pool)
    x = torch.randn(rows,8).half()
    ids = torch.tensor([[i%4,(i+1)%4] for i in range(rows)],dtype=torch.long).reshape(rows,2)
    scores = torch.full((rows,2),.5,dtype=torch.float16)
    before = x.clone()
    expected = torch.zeros_like(x)
    for expert in sorted(set(ids.reshape(-1).tolist())):
        pos,tok = torch.where((ids==expert).T)
        if expert in hot_ids:
            gu,d = bank.weights[expert];ga,up=F.linear(x[tok],gu).chunk(2,-1)
            value = F.linear(F.silu(ga)*up,d)
        else:
            value = cpu_expert(source,expert,x[tok])
        expected.index_add_(0,tok,value*scores[tok,pos,None])
    actual = dispatch(x,ids,scores)
    assert torch.equal(actual,expected) and torch.equal(x,before)
    assert dispatch.stats['hot_token_expert_pairs']+dispatch.stats['cold_token_expert_pairs']==rows*2
    assert pool.stats['peak_active_workers']<=2
    # Separate CPU-half oracle can have different accumulation rounding.
    torch.testing.assert_close(actual,routed_experts(x,ids,scores,source),atol=2e-5,rtol=4e-3)


def test_pool_really_runs_parallel_and_bounds_workers(setup):
    source,pool=setup
    barrier=threading.Barrier(2)
    class Source:
        def read(self,*args,**kwargs):
            barrier.wait(timeout=5)
            return source.read(*args,**kwargs)
    jobs=[pool.submit(Source(),e,torch.ones(1,8).half()) for e in [0,1]]
    assert all(torch.isfinite(f.result()).all() for f in jobs)
    assert pool.stats['peak_active_workers']==2
    pool.close()
    with pytest.raises(RuntimeError,match='closed'):
        pool.submit(source,0,torch.ones(1,8).half())


def test_missing_cold_expert_propagates_and_no_detached_work(setup):
    source,pool=setup
    bank=FixedHotBank(source,[0],device='cpu',dtype=torch.float16,max_bytes=288)
    dispatch=HybridExpertDispatch(bank,source,pool)
    source.experts.pop(3)
    with pytest.raises(ValueError,match='absent'):
        dispatch(torch.ones(1,8).half(),torch.tensor([[0,3]]),torch.ones(1,2).half())
    assert pool.active==0


def test_calibration_counts_tokens_not_unique_batch_experts(setup):
    source,_=setup
    counter=RouteCounter(source)
    counter(torch.ones(3,8).half(),torch.tensor([[0,1],[0,2],[0,3]]),torch.full((3,2),.5).half())
    assert counter.counts==[3,1,1,1] and counter.top(2)==(0,1)


def test_admission_and_route_validation(setup):
    source,pool=setup
    for ids,budget in [([0,0],1000),([4],1000),([0],1)]:
        with pytest.raises(ValueError):
            FixedHotBank(source,ids,device='cpu',dtype=torch.float16,max_bytes=budget)
    bank=FixedHotBank(source,[],device='cpu',dtype=torch.float16,max_bytes=0)
    dispatch=HybridExpertDispatch(bank,source,pool,max_tokens=1)
    with pytest.raises(ValueError,match='Real expert'):
        dispatch(torch.ones(1,8).half(),torch.tensor([[0,4]]),torch.ones(1,2).half())
    with pytest.raises(ValueError,match='budget'):
        dispatch(torch.ones(2,8).half(),torch.tensor([[0,1],[1,2]]),torch.ones(2,2).half())


@pytest.mark.parametrize('cpu_compute',['staged-fp32','native-half'])
def test_full_fixture_binding_close_and_default_path_unchanged(tmp_path, setup,cpu_compute):
    from test_flash_next_fp8_loader import fixture
    from test_flash_next_loader import OPTIONS
    from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
    from venturi.models.flash_next_request import FlashNextRequest
    from torch.nn.attention import sdpa_kernel, SDPBackend
    cp,_,_=fixture(tmp_path,execution_dtype=torch.float16)
    loaded=load_weight_decoded_fp8_text(cp,**dict(OPTIONS,execution_dtype=torch.float16))
    def trajectory():
        request=FlashNextRequest(loaded.core,loaded.tables[1]);out=[]
        with sdpa_kernel(SDPBackend.MATH):
            for ids in ([2,31,5],[3],[7,8]):
                p=request.preview(ids);out.append(p.logits.clone());request.commit(p)
        return out
    baseline=trajectory()
    for mode in ['fixed-hot-gpu-cold','fixed-hot-cpu-cold']:
        binding=bind_fixed_expert_policy(loaded.core,cp,mode=mode,hot_ids_per_layer=[(0,2)]*2,
            hot_bytes_per_layer=1024,workers=2,cpu_compute_mode=cpu_compute)
        actual=trajectory()
        for a,b in zip(actual,baseline):
            if mode=='fixed-hot-gpu-cold' or cpu_compute=='native-half':
                assert torch.equal(a,b)
            else:
                torch.testing.assert_close(a,b,atol=.004,rtol=.01)
        binding.close()
        assert all(layer.mlp.expert_dispatch is None for layer in loaded.core.layers)
        assert all(torch.equal(a,b) for a,b in zip(trajectory(),baseline))


def test_cpu_cache_budget_rounding_eviction_and_live_lease(setup):
    source,_=setup
    cache=CPUDecodedExpertCache(source,max_bytes=576)
    first=cache.read_compute(0)
    assert cache.read_compute(0) is first
    assert all(torch.equal(w,original.float()) for w,original in zip(first,source.read(0,device='cpu',dtype=torch.float16)))
    cache.read_compute(1)
    assert cache.stats['evictions']==1 and cache.stats['retained_bytes']==576
    assert torch.equal(first[0],source.experts[0][0].float())
    cache.clear()
    assert cache.stats['retained_bytes']==0
    uncached=CPUDecodedExpertCache(source,max_bytes=0)
    x=torch.ones(2,8).half()
    assert torch.equal(cpu_expert(cache,0,x),cpu_expert(uncached,0,x))
    assert uncached.stats['bypasses']==1 and uncached.stats['entries']==0


def test_cpu_cache_distinct_misses_parallel_and_same_key_coalesces(setup):
    from concurrent.futures import ThreadPoolExecutor
    import time
    source,_=setup
    barrier=threading.Barrier(2)
    class Source:
        geometry=source.geometry
        def read(self,*args,**kwargs):
            barrier.wait(timeout=5)
            return source.read(*args,**kwargs)
    cache=CPUDecodedExpertCache(Source(),max_bytes=1152)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a,b=[pool.submit(cache.read_compute,e) for e in [0,1]]
        assert a.result()[0].dtype==b.result()[0].dtype==torch.float32
    started,release=threading.Event(),threading.Event()
    class Slow:
        geometry=source.geometry
        def read(self,*args,**kwargs):
            started.set();assert release.wait(timeout=5)
            return source.read(*args,**kwargs)
    cache=CPUDecodedExpertCache(Slow(),max_bytes=576)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(cache.read_compute,0);assert started.wait(5)
        second=pool.submit(cache.read_compute,0)
        for _ in range(100):
            if cache.stats['coalesced']:break
            time.sleep(.001)
        release.set();assert first.result() is second.result()
    assert cache.stats['misses']==1 and cache.stats['coalesced']==1
