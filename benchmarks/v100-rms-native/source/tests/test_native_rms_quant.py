"""Native post-norm packing shares no stale scratch and preserves both TP ranks."""
import copy
import ctypes as C
import os

import pytest
import torch
from test_qwen_model import hybrid_fixture
from test_qwen_source import write_source
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenModel,QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend
from venturi.qwen_export import export_qwen_q8


@pytest.fixture(scope='module')
def libraries():
    paths=[os.environ.get('VENTURI_RMS_BASELINE'),os.environ.get('VENTURI_RMS_NATIVE')]
    if not all(paths):pytest.skip('Explicit native RMS OFF/ON libraries and two V100s required')
    result=[CudaBackend(p) for p in paths]
    for enabled,b in enumerate(result):
        n=b._native;n.venturi_q8_rms_quant_enabled.argtypes=[];n.venturi_q8_rms_quant_enabled.restype=C.c_int
        n.venturi_q8_rms_quant_eligible.argtypes=[C.c_int,C.c_int];n.venturi_q8_rms_quant_eligible.restype=C.c_int
        assert n.venturi_q8_rms_quant_enabled()==enabled
    return result


@pytest.fixture(scope='module')
def wide(tmp_path_factory):
    root=tmp_path_factory.mktemp('rms-eligible');old,raw=hybrid_fixture(root)
    cfg=copy.deepcopy(old.config);cfg['hidden_size']=5120
    g=torch.Generator().manual_seed(40149)
    for name,v in list(raw.items()):
        shape=list(v.shape)
        if name.endswith(('embed_tokens.weight','lm_head.weight')):shape[-1]=5120
        elif name.endswith(('input_layernorm.weight','post_attention_layernorm.weight')) or name=='model.language_model.norm.weight':shape=[5120]
        elif name.endswith(('mlp.gate_proj.weight','mlp.up_proj.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','in_proj_qkv.weight','in_proj_z.weight','in_proj_a.weight','in_proj_b.weight')):shape[-1]=5120
        elif name.endswith(('mlp.down_proj.weight','self_attn.o_proj.weight','linear_attn.out_proj.weight')):shape[0]=5120
        if tuple(shape)!=v.shape:raw[name]=(torch.randn(shape,generator=g)*(0.004 if len(shape)>1 else 0.03)).to(v.dtype)
    source=write_source(root/'rms-wide-source',raw=dict(model_type='qwen3_5',text_config=cfg),tensors=raw)
    return QwenQ8Weights.from_ninfer(NInferArtifact(export_qwen_q8(source,root/'wide.ninfer')['output']))


def load(weights,b,rows=33):
    return NativeQwenModel.from_q8(weights,b,weight_mode='resident',dtype=torch.float16,device='cuda:0',max_tokens=rows,reserve_bytes=0,
        tensor_parallel_device='cuda:1',tensor_parallel_transport='host',projection_profile='q8-residual-column-mlp')


def equal(a,b):assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def test_native_rms_guard_and_default(libraries):
    for enabled,b in enumerate(libraries):
        for t in [-1,0,1,2,8,23,32,33]:
            for k in [32,5120,8704]:assert b._native.venturi_q8_rms_quant_eligible(t,k)==int(enabled and 1<=t<=32 and k==5120)


@pytest.mark.parametrize('tokens',[1,2,8,23,32,33])
def test_eligible_norm_full_mlp_both_rank_bytes(wide,libraries,tokens):
    models=[load(wide,b) for b in libraries];sessions=[m.create_session(64) for m in models]
    x=(torch.randn(tokens,5120,generator=torch.Generator().manual_seed(tokens+73))*0.2).half()
    try:
        results=[]
        for enabled,m,s in zip([False,True],models,sessions):
            assert m.projection_info['rms_quant_fusion']['model_has_eligible_norm']==enabled
            fn=m._native.venturi_qwen_rms_mlp_debug;fn.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int]+[C.c_void_p]*4;fn.restype=C.c_int
            out=torch.empty_like(x);normalized=torch.empty_like(x);activation=torch.empty(tokens,64,dtype=torch.float16);partials=torch.empty(2,tokens,5120,dtype=torch.float32)
            rows=[]
            for layer in [0,1]:
                m._backend._check(fn(s._handle,layer,x.data_ptr(),tokens,out.data_ptr(),normalized.data_ptr(),activation.data_ptr(),partials.data_ptr()))
                rows.append([v.clone() for v in [out,normalized,activation,partials]])
            results.append(rows)
        for a,b in zip(*results):
            for u,v in zip(a,b):equal(u,v)
    finally:
        for s in sessions:s.close()
        for m in models:m.close()


def test_eligible_hybrid_traces_state_mixed_cohort_reset_and_failure(wide,libraries):
    models=[load(wide,b,8) for b in libraries];sessions=[m.create_session(64) for m in models];batches=[]
    try:
        for ids in [[1,2,3,4,5,6,7,8],[9],[3,6]]:
            traces=[[],[]];logits=[s.forward(ids,all_logits=True,trace=trace) for s,trace in zip(sessions,traces)]
            equal(*logits);same_state(*[s.snapshot() for s in sessions])
            for a,b in zip(*traces):equal(a['hidden'],b['hidden'])
        for s in sessions:s.reset()
        ids=[s.generate([1,2,3],max_new_tokens=8) for s in sessions];assert ids[0]==ids[1];same_state(*[s.snapshot() for s in sessions])
        batches=[m.create_batch([48]*8) for m in models]
        prompts=[[1+j%10,2,3] if j%2 else [4,5] for j in range(8)]
        kwargs=dict(max_new_tokens=[1+j%5 for j in range(8)],eos_token_ids=[[ids[0][0]] if j%2 else [] for j in range(8)])
        result=[b.generate(prompts,**kwargs) for b in batches];assert result[0]==result[1]
        for j in range(8):same_state(*[b.snapshot(j) for b in batches])
        before=[batches[1].snapshot(j) for j in range(8)]
        for b in batches:b.reset(slots=[0,7]);b.forward([[6],[2,4]],slots=[0,7])
        for j in range(8):same_state(*[b.snapshot(j) for b in batches])
        for j in range(1,7):same_state(before[j],batches[1].snapshot(j))
        for b in batches:
            b.reset(slots=[0,7]);b._inject_failure(slot=7,layer=2,position=0)
            with pytest.raises(RuntimeError,match='injected'):
                b.generate([[2,4],[6]],slots=[0,7],max_new_tokens=3)
            b.reset(slots=[0,7])
        results=[b.generate([[2,4],[6]],slots=[0,7],max_new_tokens=3) for b in batches]
        assert results[0]==results[1]
        for j in range(8):same_state(*[b.snapshot(j) for b in batches])
    finally:
        for b in batches:b.close()
        for s in sessions:s.close()
        for m in models:m.close()
