"""VRAM-first KV is opt-in, exact-byte, pressure-driven and model-wide."""
import ctypes as C
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
import torch
from test_native_head_gdn import eligible_mtp, PROFILE
from test_native_experiments import exact
from venturi.models import NativeQwenModel, QwenQ8Weights, NativeQwenMTPModel
from venturi.ops import CudaBackend


def test_validation_before_cuda():
    w,b=object.__new__(QwenQ8Weights),object.__new__(CudaBackend)
    b._native=SimpleNamespace()
    kw=dict(weight_mode='resident',kv_cache='vram-first')
    for values in [dict(kv_cache='ram-first'),dict(kv_gpu_budget_bytes=-1),dict(kv_ram_budget_bytes=True),dict(kv_ram_budget_bytes=1.2)]:
        with pytest.raises(ValueError):NativeQwenModel.from_q8(w,b,**(kw|values))
    with pytest.raises(ValueError,match='budgets require'):
        NativeQwenModel.from_q8(w,b,weight_mode='resident',kv_gpu_budget_bytes=1)
    with pytest.raises(ValueError,match='complete resident'):
        NativeQwenModel.from_q8(w,b,**(kw|dict(weight_mode='streaming')))
    with pytest.raises(RuntimeError,match='exports absent'):
        NativeQwenModel.from_q8(w,b,**kw)


@pytest.fixture
def engines(tmp_path):
    path=os.environ.get('VENTURI_KV_TIER_LIBRARY')
    if not path:pytest.skip('Explicit SM70 library and two V100s required')
    w,draft,_,_=eligible_mtp(tmp_path/'source');backend=CudaBackend(path)
    def load(**kw):
        return NativeQwenModel.from_q8(w,backend,weight_mode='resident',dtype=torch.float16,device='cuda:0',
            max_tokens=32,reserve_bytes=0,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',projection_profile=PROFILE,**kw)
    load.attention_layers=w.config['layer_types'].count('full_attention')
    probe=load(kv_cache='vram-first');g=probe.kv_cache_info()['page_bytes'];probe.close()
    return load,g,draft


@pytest.mark.parametrize('pressure,mask',[(False,0),(True,0),(True,15)])
def test_exact_scalar_batch_reset_reorder_eos(engines,pressure,mask):
    load,g,_=engines;a=load();b=load(kv_cache='vram-first',kv_gpu_budget_bytes=2*g if pressure else 0,kv_ram_budget_bytes=256*g,experimental_flags=mask)
    try:
        assert b.kv_cache_info()['gpu_pool_bytes']==b.kv_cache_info()['cpu_backing_bytes']==0
        x,y=a.create_session(64),b.create_session(64)
        try:
            for ids in ([1,2,3],[4],[5,6]):
                exact(x.forward(ids,all_logits=True),y.forward(ids,all_logits=True));exact(x.snapshot(),y.snapshot())
        finally:x.close();y.close()
        x,y=a.create_batch([64]*8),b.create_batch([64]*8)
        try:
            for slots in (list(range(8)),[7,5,2,0],list(reversed(range(8)))):
                ids=[[1+i%3,2] for i in slots]
                exact(x.forward(ids,slots=slots),y.forward(ids,slots=slots))
                for i in slots:exact(x.snapshot(i),y.snapshot(i))
            first=b.kv_cache_info();[y.snapshot(i) for i in range(8)];assert b.kv_cache_info()==first
            if pressure:assert first['evictions']>0 and first['reloads']>0 and first['cpu_backing_bytes']>0
            else:assert first['evictions']==first['reloads']==first['cpu_backing_bytes']==0
            x.reset();y.reset();assert b.kv_cache_info()['cpu_backing_bytes']==0
            prompts=[[1,2,3+i%3] for i in range(8)]
            expected=x.generate(prompts,max_new_tokens=6,eos_token_ids=[])
            exact(expected,y.generate(prompts,max_new_tokens=6,eos_token_ids=[]))
            x.reset();y.reset();eos=[[v[0]] if i%2 else [] for i,v in enumerate(expected)]
            exact(x.generate(prompts,max_new_tokens=6,eos_token_ids=eos),y.generate(prompts,max_new_tokens=6,eos_token_ids=eos))
            x.reset();y.reset();y._inject_failure(slot=0,layer=0,position=0)
            with pytest.raises(RuntimeError):y.forward(prompts)
            y.reset();exact(x.forward(prompts),y.forward(prompts))
        finally:x.close();y.close()
        assert b.kv_cache_info()['entries']==0 and b.kv_cache_info()['cpu_backing_bytes']==0
    finally:a.close();b.close()


def test_quota_failure_no_scalar_mutation_and_mtp_rejected(engines):
    load,g,draft=engines;m=load(kv_cache='vram-first',kv_gpu_budget_bytes=g,kv_ram_budget_bytes=0)
    s=m.create_session(64)
    try:
        with pytest.raises(RuntimeError,match='cannot fit'):s.forward([1,2])
        assert s.info()['position']==0 and not s.info()['failed']
        with pytest.raises(ValueError,match='not qualified'):NativeQwenMTPModel.from_q8(m,draft)
    finally:s.close();m.close()
    m=load(kv_cache='vram-first',kv_gpu_budget_bytes=2*g*load.attention_layers,kv_ram_budget_bytes=0);a=m.create_session(64);b=m.create_session(64)
    try:
        a.forward([1,2]);before=a.snapshot()
        with pytest.raises(RuntimeError,match='RAM budget'):b.forward([3])
        assert b.info()['position']==0 and not b.info()['failed'];exact(before,a.snapshot())
    finally:a.close();b.close();m.close()


def test_close_cold_cohort_during_decode(engines):
    load,g,_=engines;r=load();m=load(kv_cache='vram-first',kv_gpu_budget_bytes=2*g,kv_ram_budget_bytes=256*g)
    a=m.create_batch([64]*2);b=m.create_batch([64]*2);c=r.create_batch([64]*2)
    try:
        a.forward([[1],[2]]);exact(c.forward([[3],[4]]),b.forward([[3],[4]]))
        def decode():
            for _ in range(12):b.forward([[5],[6]])
        with ThreadPoolExecutor(max_workers=1) as p:
            f=p.submit(decode);a.close();f.result(timeout=30)
        for _ in range(12):c.forward([[5],[6]])
        for i in range(2):exact(c.snapshot(i),b.snapshot(i))
    finally:a.close();b.close();c.close();r.close();m.close()

@pytest.mark.parametrize('args,match',[
    (['--kv-cache','vram-first','--weight-mode','streaming'],'resident single/TP'),
    (['--kv-cache','vram-first','--pipeline-device','cuda:1'],'not PP/MTP'),
    (['--kv-gpu-budget-mib','10'],'require --kv-cache'),
    (['--kv-cache','vram-first','--kv-ram-budget-mib','-1'],'nonnegative'),
])
def test_cli_rejects_invalid_scope_before_backend(tmp_path,monkeypatch,capsys,args,match):
    from test_qwen_model import hybrid_fixture
    from venturi import qwen_native_cli as cli
    w,_=hybrid_fixture(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('Backend must not be initialized')
    monkeypatch.setattr(cli,'CudaBackend',forbidden)
    with pytest.raises(SystemExit):cli.main(['generate',str(w.artifact.path),'--weight-mode','resident','--token-ids','1','--blocks-library','not-loaded.so']+args)
    assert match in capsys.readouterr().err


def test_closed_owner_stats_and_query_close_race_without_cuda():
    import threading
    class Finalizer:
        alive=True
        def __call__(self):self.alive=False
    m=object.__new__(NativeQwenModel);m._lock=threading.RLock();m._finalizer=Finalizer();m._kv_cache_mode='vram-first';m._handle=123
    def query(handle,out):
        assert m._finalizer.alive,'query used a freed model handle'
        for i in range(18):out[i]=i
        return 0
    m._native=SimpleNamespace(venturi_qwen_kv_tier_info=query);m._backend=SimpleNamespace(_check=lambda r:None)
    def reader():
        for _ in range(1000):
            r=m._kv_cache_info_for_request();assert r['mode']=='vram-first'
    with ThreadPoolExecutor(max_workers=2) as pool:
        f=pool.submit(reader);m.close();f.result(timeout=10)
    assert m._kv_cache_info_for_request()['model_owner_closed']
    with pytest.raises(RuntimeError,match='model closed'):m.kv_cache_info()


def test_sessions_and_batch_outlive_model_owner(engines):
    load,g,_=engines;r=load();m=load(kv_cache='vram-first',kv_gpu_budget_bytes=2*g,kv_ram_budget_bytes=256*g)
    a,b=r.create_session(64),m.create_session(64);c,d=r.create_batch([64]*2),m.create_batch([64]*2)
    try:
        m.close();m.close()
        with pytest.raises(RuntimeError,match='model closed'):m.create_session(64)
        exact(a.forward([1,2]),b.forward([1,2]));exact(a.snapshot(),b.snapshot())
        exact(c.forward([[3],[4]]),d.forward([[3],[4]]));exact(c.generate([[5],[6]],max_new_tokens=4,eos_token_ids=[]),d.generate([[5],[6]],max_new_tokens=4,eos_token_ids=[]))
        assert b.info()['shared_model_kv_cache']['model_owner_closed']
        for i in range(2):exact(c.snapshot(i),d.snapshot(i))
    finally:a.close();b.close();c.close();d.close();r.close();m.close()
