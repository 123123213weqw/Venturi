"""Block decoder/store contracts, no activation quantization/model speed claim."""

from dataclasses import replace
import json
from types import MappingProxyType

import pytest
from safetensors.torch import save_file
import torch

from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore, decode_block_fp8
from venturi.models.flash_next_moe import MoEGeometry


@pytest.mark.parametrize('strategy', ['tiled', 'vectorized'])
@pytest.mark.parametrize('shape', [(1, 1), (127, 129), (256, 128), (129, 257)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_block_decoder_matches_independent_scale_expansion(shape, dtype, strategy):
    m, n = shape
    codes = (torch.arange(m*n).reshape(shape)%63-31).float().to(torch.float8_e4m3fn)
    scales = (torch.arange(((m+127)//128)*((n+127)//128)).reshape((m+127)//128, (n+127)//128)*.003+.007).bfloat16()
    actual = decode_block_fp8(codes, scales, dtype=dtype, strategy=strategy)
    expanded = scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:m, :n]
    expected = (codes.float()*expanded).to(dtype)
    assert torch.equal(actual, expected) and actual.is_contiguous()


def fixture(path):
    g = MoEGeometry(8, 2, 1, 6, 6)
    weights = {}
    for e in range(2):
        for p, shape in [('gate_proj', (6, 8)), ('up_proj', (6, 8)), ('down_proj', (8, 6))]:
            name = f'mlp.experts.{e}.{p}'
            weights[name+'.weight'] = (torch.arange(48).reshape(shape)%15-7).float().to(torch.float8_e4m3fn)
            weights[name+'.weight_scale_inv'] = torch.tensor([[.013+e*.002]], dtype=torch.bfloat16)
    save_file(weights, path/'model.safetensors')
    (path/'config.json').write_text(json.dumps(dict(quantization_config=dict(quant_method='fp8',
        weight_block_size=[128, 128], weight_per_tensor=False, activation_scheme='dynamic'))))
    return inspect_checkpoint(path), g, weights


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_store_separate_gate_up_ownership_dtype_and_selected_expert(tmp_path, dtype):
    cp, g, weights = fixture(tmp_path)
    store = BlockFP8ExpertStore(cp, g, 'mlp.experts')
    gu, down = store.read(1, device='cpu', dtype=dtype)
    expected = [(weights[f'mlp.experts.1.{p}.weight'].float()*weights[f'mlp.experts.1.{p}.weight_scale_inv'].float()).to(dtype)
                for p in ('gate_proj', 'up_proj', 'down_proj')]
    assert torch.equal(gu, torch.cat(expected[:2])) and torch.equal(down, expected[-1])
    gu.zero_()
    assert store.read(1, device='cpu', dtype=dtype)[0].any()
    with pytest.raises(ValueError):
        store.read(True, device='cpu', dtype=dtype)


def test_invalid_encoding_budgets_missing_expert_and_nonfinite_rejected(tmp_path):
    cp, g, _ = fixture(tmp_path)
    for kwargs in (dict(max_serialized_expert_bytes=1), dict(max_decoded_expert_bytes=1)):
        with pytest.raises(ValueError, match='budget'):
            BlockFP8ExpertStore(cp, g, 'mlp.experts', **kwargs)
    mapping = dict(cp.tensors)
    mapping.pop('mlp.experts.1.up_proj.weight')
    with pytest.raises(ValueError, match='required'):
        BlockFP8ExpertStore(replace(cp, tensors=MappingProxyType(mapping)), g, 'mlp.experts')
    raw = dict(cp.config)
    raw['quantization_config'] = dict(raw['quantization_config'], weight_block_size=[32, 32])
    with pytest.raises(ValueError, match='encoding'):
        BlockFP8ExpertStore(replace(cp, config=raw), g, 'mlp.experts')
    codes, scales = torch.ones(1, 1).to(torch.float8_e4m3fn), torch.ones(1, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='budget'):
        decode_block_fp8(codes, scales, dtype=torch.float32, max_output_bytes=1)
    with pytest.raises(ValueError, match='positive finite'):
        decode_block_fp8(codes, scales*0, dtype=torch.float32)
    with pytest.raises(ValueError, match='Nonfinite'):
        decode_block_fp8(torch.full((1, 1), float('nan')).to(torch.float8_e4m3fn), scales, dtype=torch.float32)
    with pytest.raises(ValueError, match='overflows'):
        decode_block_fp8(codes, torch.full((1, 1), 1e10).bfloat16(), dtype=torch.float16)


def test_vectorized_work_admission_and_tiled_tail_fallback():
    codes = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scales = torch.ones(1, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='work byte budget'):
        decode_block_fp8(codes, scales, dtype=torch.float32, strategy='vectorized', max_work_bytes=1)
    tail = decode_block_fp8(codes[:1, :1], scales, dtype=torch.float32, strategy='vectorized', max_work_bytes=1)
    assert tail.item() == 1
