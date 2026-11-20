"""Qualified-size fusion actually executes in small native hybrid fixtures."""
import copy
import ctypes as C
import os

import pytest
import torch

from test_qwen_model import hybrid_fixture
from test_qwen_source import write_source
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend
from venturi.qwen_export import export_qwen_q8


@pytest.fixture(scope='module')
def libraries():
    paths=[os.environ.get('VENTURI_SWIGLU_BASELINE'),os.environ.get('VENTURI_SWIGLU_NATIVE')]
    if not all(paths):pytest.skip('Explicit OFF/ON SM70 libraries and two V100s required')
    result=[CudaBackend(p) for p in paths]
    for expected,b in enumerate(result):
        n=b._native
        n.venturi_q8_swiglu_quant_enabled.argtypes=[];n.venturi_q8_swiglu_quant_enabled.restype=C.c_int
        n.venturi_q8_swiglu_quant_eligible.argtypes=[C.c_int,C.c_int];n.venturi_q8_swiglu_quant_eligible.restype=C.c_int
        assert n.venturi_q8_swiglu_quant_enabled()==expected
    return result


def wide_fixture(root):
    original,raw=hybrid_fixture(root)
    cfg=copy.deepcopy(original.config);cfg['intermediate_size']=17408
    g=torch.Generator().manual_seed(4012)
    for name,value in list(raw.items()):
        if name.endswith(('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight')):
            shape=(32,17408) if name.endswith('mlp.down_proj.weight') else (17408,32)
            raw[name]=(torch.randn(shape,generator=g)*0.025).bfloat16()
    source=write_source(root/'wide-source',raw=dict(model_type='qwen3_5',text_config=cfg),tensors=raw)
    return QwenQ8Weights.from_ninfer(NInferArtifact(export_qwen_q8(source,root/'wide.ninfer')['output']))


def load(weights,b,max_tokens=32):
    return NativeQwenModel.from_q8(weights,b,weight_mode='resident',dtype=torch.float16,device='cuda:0',
        max_tokens=max_tokens,reserve_bytes=0,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',
        projection_profile='q8-residual-column-mlp')


def same(a,b):
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def test_native_routing_guard_and_default(libraries):
    for enabled,b in enumerate(libraries):
        for t in [-1,0,1,2,7,8,23,32,33]:
            for k in [32,67,5120,8704,17408]:
                assert b._native.venturi_q8_swiglu_quant_eligible(t,k)==int(enabled and 1<=t<=32 and k==8704)


@pytest.mark.parametrize('tokens',[1,2,8,23,32,33])
def test_eligible_full_native_mlp_debug_bytes_and_fallback(tmp_path,libraries,tokens):
    weights=wide_fixture(tmp_path)
    models=[load(weights,b,33) for b in libraries];sessions=[m.create_session(48) for m in models]
    gen=torch.Generator().manual_seed(61+tokens);x=(torch.randn(tokens,32,generator=gen)*0.2).half()
    try:
        got=[]
        for m,s in zip(models,sessions):
            assert m.projection_info['swiglu_quant_fusion']['model_has_eligible_mlp']==bool(m is models[1])
            out=torch.empty(tokens,32,dtype=torch.float16);act=torch.empty(tokens,17408,dtype=torch.float16)
            part=torch.empty(2,tokens,32,dtype=torch.float32)
            fn=m._native.venturi_qwen_column_mlp_debug
            fn.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int]+[C.c_void_p]*3;fn.restype=C.c_int
            rows=[]
            for layer in [0,1]:
                m._backend._check(fn(s._handle,layer,x.data_ptr(),tokens,out.data_ptr(),act.data_ptr(),part.data_ptr()))
                rows.append([v.clone() for v in (out,act,part)])
            got.append(rows)
        for a,b in zip(*got):
            for u,v in zip(a,b):same(u,v)
    finally:
        for s in sessions:s.close()
        for m in models:m.close()


def test_eligible_native_trajectory_traces_state_reset_and_private_cohort(tmp_path,libraries):
    weights=wide_fixture(tmp_path);models=[load(weights,b,8) for b in libraries]
    sessions=[m.create_session(64) for m in models]
    batches=[]
    try:
        for ids in ([1,2,3,4,5,6,7,8],[9],[3,6]):
            traces=[[],[]]
            logits=[s.forward(ids,all_logits=True,trace=trace) for s,trace in zip(sessions,traces)]
            same(*logits)
            for a,b in zip(*traces):same(a['hidden'],b['hidden'])
            same_state(*[s.snapshot() for s in sessions]);assert sessions[0].position==sessions[1].position
        for s in sessions:s.reset()
        tokens=[s.generate([1,2,3],max_new_tokens=8) for s in sessions]
        assert tokens[0]==tokens[1];same_state(*[s.snapshot() for s in sessions])
        batches=[m.create_batch([48]*8) for m in models]
        prompts=[[1+j%10,2,3] if j%2 else [4,5] for j in range(8)]
        kwargs=dict(max_new_tokens=[1+j%5 for j in range(8)],eos_token_ids=[[tokens[0][0]] if j%2 else [] for j in range(8)])
        result=[b.generate(prompts,**kwargs) for b in batches];assert result[0]==result[1]
        for j in range(8):same_state(*[b.snapshot(j) for b in batches])
        before=[batches[1].snapshot(j) for j in range(8)]
        for b in batches:b.reset(slots=[0,7]);b.forward([[6],[2,4]],slots=[0,7])
        for j in range(8):same_state(*[b.snapshot(j) for b in batches])
        for j in range(1,7):same_state(before[j],batches[1].snapshot(j))
    finally:
        for b in batches:b.close()
        for s in sessions:s.close()
        for m in models:m.close()
