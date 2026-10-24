"""Opt-in SM70 experiments: exact bytes, private state and graph lifetimes."""
import ctypes as C
import os
import pytest
import torch
from types import SimpleNamespace
from test_native_head_gdn import eligible_mtp, PROFILE
from venturi.models import NativeQwenModel,QwenQ8Weights
from venturi.ops import CudaBackend

def test_experiment_validation_before_cuda():
    w,b=object.__new__(QwenQ8Weights),object.__new__(CudaBackend)
    b._native=SimpleNamespace()
    for mask in (-1,16,True,1.5):
        with pytest.raises(ValueError,match='integer mask'):
            NativeQwenModel.from_q8(w,b,weight_mode='resident',experimental_flags=mask)
    with pytest.raises(ValueError,match='head-GDN'):
        NativeQwenModel.from_q8(w,b,weight_mode='resident',experimental_flags=1)
    with pytest.raises(RuntimeError,match='exports absent'):
        NativeQwenModel.from_q8(w,b,weight_mode='resident',dtype=torch.float16,projection_profile=PROFILE,
            tensor_parallel_device='cuda:1',tensor_parallel_transport='host',experimental_flags=1)

@pytest.fixture
def setup(tmp_path):
    path=os.environ.get('VENTURI_EXPERIMENT_LIBRARY')
    if not path:pytest.skip('Explicit SM70 library/two V100s required')
    w,_,_,_=eligible_mtp(tmp_path/'source')
    backend=CudaBackend(path)
    def load(mask):
        return NativeQwenModel.from_q8(w,backend,weight_mode='resident',dtype=torch.float16,device='cuda:0',
            max_tokens=32,reserve_bytes=0,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',
            projection_profile=PROFILE,experimental_flags=mask)
    return load

def exact(a,b):
    if torch.is_tensor(a):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

def info(m):
    f=m._native.venturi_qwen_experiment_info;f.argtypes=[C.c_void_p,C.POINTER(C.c_uint64)];f.restype=C.c_int
    a=(C.c_uint64*4)();m._backend._check(f(m._handle,a));return list(a)

@pytest.mark.parametrize('mask',[1,2,4,8,3,7,15])
def test_exact_experiments(setup,mask):
    baseline,candidate=setup(0),setup(mask)
    try:
        for count in (1,2,8):
            a,b=baseline.create_batch([64]*count),candidate.create_batch([64]*count)
            try:
                prompts=[[1,2,3+i%3] for i in range(count)]
                for step in range(5):
                    inputs=prompts if not step else [[4+i] for i in range(count)]
                    exact(a.forward(inputs),b.forward(inputs))
                    for slot in range(count):exact(a.snapshot(slot),b.snapshot(slot))
                if mask&4:
                    x=info(candidate);assert x[1]>0 and x[3]>x[2]
                a.reset();b.reset()
                for slots in ([count-1],list(reversed(range(count)))):
                    inputs=[[5,6] for _ in slots]
                    exact(a.forward(inputs,slots=slots),b.forward(inputs,slots=slots))
                    exact(a.forward([[7] for _ in slots],slots=slots),b.forward([[7] for _ in slots],slots=slots))
                a.reset();b.reset()
                expected=a.generate(prompts,max_new_tokens=7,eos_token_ids=[])
                exact(expected,b.generate(prompts,max_new_tokens=7,eos_token_ids=[]))
                a.reset();b.reset()
                eos=[[expected[i][0]] if i%2 else [] for i in range(count)]
                exact(a.generate(prompts,max_new_tokens=7,eos_token_ids=eos),b.generate(prompts,max_new_tokens=7,eos_token_ids=eos))
                a.reset();b.reset()
                b._inject_failure(slot=0,layer=0,position=0)
                with pytest.raises(RuntimeError):b.forward(prompts)
                b.reset()
                exact(a.forward(prompts),b.forward(prompts))
            finally:a.close();b.close()
            if mask&4:assert info(candidate)[1]==0 # no request pointer can outlive its owner
    finally:baseline.close();candidate.close()

def test_mtp_independent_export_verification(tmp_path):
    import json
    from venturi.qwen_export_verify import verify_qwen_q8_export
    target,draft,_,source=eligible_mtp(tmp_path/'source')
    from venturi.qwen_export import export_qwen_mtp_q8
    path=tmp_path/'verify.ninfer';report=export_qwen_mtp_q8(source,path)
    r=tmp_path/'report.json';r.write_text(json.dumps(report))
    check=verify_qwen_q8_export(source,path,r,component='mtp',source_manifest=str(source/'venturi-source.json'))
    assert check['component']=='mtp' and check['component_tensors']==15
    assert check['full_floating_tensors_equal']==7 and check['independently_checked_q8_rows']==24
    with pytest.raises(ValueError,match='Explicit component'):
        verify_qwen_q8_export(source,path,r,component='wrong')

def test_graph_close_other_cohort_while_decoding(setup):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    reference,candidate=setup(0),setup(4)
    a=candidate.create_batch([64]*2);b=candidate.create_batch([64]*2);r=reference.create_batch([64]*2)
    ready=threading.Event()
    try:
        a.forward([[1],[2]])
        exact(r.forward([[3],[4]]),b.forward([[3],[4]]))
        def decode():
            ready.set()
            for _ in range(24):b.forward([[5],[6]])
        with ThreadPoolExecutor(max_workers=1) as pool:
            f=pool.submit(decode);assert ready.wait(5);a.close();f.result(timeout=20)
        for _ in range(24):r.forward([[5],[6]])
        for i in range(2):exact(r.snapshot(i),b.snapshot(i))
    finally:a.close();b.close();r.close();reference.close();candidate.close()
