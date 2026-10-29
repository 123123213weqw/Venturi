"""Complete immutable tiny Q4 export, source-independent reload and fail-closed IO."""
import hashlib
import json

import pytest
from safetensors import safe_open
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from test_flash_next_fp8_loader import fixture
from test_flash_next_loader import OPTIONS
from test_flash_next_text import assert_states_equal
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text,load_weight_decoded_q4_text
from venturi.models.flash_next_moe import TensorExpertStore,MoEGeometry
from venturi.models.flash_next_q4 import quantize_q4_rtn,decode_q4_fp16
from venturi.models.flash_next_q4_bundle import export_q4_text,bundle_manifest,CompleteQ4ExpertStore
from venturi.models.flash_next_request import FlashNextRequest

IDENTITY=dict(revision='fixture-original-FP8',manifest_sha256='fixture-authenticated',decode_dtype='FP16')


def make(tmp_path):
    cp,_,_=fixture(tmp_path/'source',execution_dtype=torch.float16)
    export_q4_text(cp,tmp_path/'bundle',source_identity=IDENTITY,workers=2)
    return cp,tmp_path/'bundle'


def test_full_export_reload_without_original_weights_and_exact_states(tmp_path):
    cp,path=make(tmp_path);m=bundle_manifest(path,verify_payloads=True)
    assert m['expert_count']==8 and m['layers']==2
    original=load_weight_decoded_fp8_text(cp,**dict(OPTIONS,execution_dtype=torch.float16))
    g=MoEGeometry.from_dict(dict(cp.config))
    for i,layer in enumerate(original.core.layers):
        s=BlockFP8ExpertStore(cp,g,f'model.language_model.layers.{i}.mlp.experts')
        layer.mlp.expert_store=TensorExpertStore(g,{e:tuple(decode_q4_fp16(quantize_q4_rtn(w)) for w in s.read(e,device='cpu',dtype=torch.float16)) for e in range(g.num_experts)})
    bundle=inspect_checkpoint(path)
    assert not any('.mlp.experts.' in n for n in bundle.tensors)
    loaded=load_weight_decoded_q4_text(bundle,**OPTIONS)
    # Move original after its PLE reader has opened immutable multipliers/scales;
    # compare requests using the new bundle reader on BOTH independent cores.
    cp.directory.rename(tmp_path/'source-unavailable')
    with sdpa_kernel(SDPBackend.MATH):
        a,b=FlashNextRequest(loaded.core,loaded.tables[1]),FlashNextRequest(original.core,loaded.tables[1])
        for ids in ([2,31,5],[3],[7,8]):
            x,y=a.preview(ids),b.preview(ids);assert torch.equal(x.logits,y.logits)
            a.commit(x);b.commit(y);assert_states_equal(a.state,b.state)
    assert all(isinstance(layer.mlp.expert_store,CompleteQ4ExpertStore) for layer in loaded.core.layers)
    assert all(layer.mlp.expert_store.converted==0 for layer in loaded.core.layers)
    with pytest.raises(ValueError):
        load_weight_decoded_fp8_text(bundle,**OPTIONS)


def test_raw_core_ple_bytes_preserved_and_no_experts_in_body(tmp_path):
    cp,path=make(tmp_path);new=inspect_checkpoint(path)
    for name,spec in new.tensors.items():
        assert '.mlp.experts.' not in name
        with safe_open(str(spec.file),framework='pt',device='cpu') as a,safe_open(str(cp.tensors[name].file),framework='pt',device='cpu') as b:
            x,y=a.get_tensor(name),b.get_tensor(name)
            assert x.dtype==y.dtype and torch.equal(x.view(torch.uint8),y.view(torch.uint8))


def test_complete_bundle_missing_and_corrupt_payload_reject(tmp_path):
    _,path=make(tmp_path)
    f=path/'experts/layer-00/expert-0000.safetensors'
    raw=f.read_bytes();f.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError,match='checksum'):
        bundle_manifest(path,verify_payloads=True)
    # Lazy first-read payload checksum must also detect same-size corruption.
    loaded=load_weight_decoded_q4_text(inspect_checkpoint(path),**OPTIONS)
    with pytest.raises(ValueError,match='checksum'):
        loaded.core.layers[0].mlp.expert_store.packed(0)
    f.unlink()
    with pytest.raises(ValueError,match='Missing'):
        bundle_manifest(path)


def test_no_marker_no_conversion_and_wrong_precision(tmp_path):
    _,path=make(tmp_path)
    with pytest.raises(ValueError,match='FP16'):
        load_weight_decoded_q4_text(inspect_checkpoint(path),**dict(OPTIONS,execution_dtype=torch.float32))
    (path/'.venturi-q4-complete.json').unlink()
    with pytest.raises(FileNotFoundError):bundle_manifest(path)
    with pytest.raises(FileExistsError):
        export_q4_text(inspect_checkpoint(path),path,source_identity=IDENTITY)
