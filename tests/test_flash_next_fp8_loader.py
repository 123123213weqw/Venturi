"""Explicit FP8-weight -> floating reference, NOT a W8A8 engine equivalence."""

from dataclasses import replace
import json
from types import MappingProxyType

import pytest
from safetensors.torch import load_file, save_file
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next_loader import fixture as floating_fixture, OPTIONS
from test_flash_next_text import assert_states_equal
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_loader import (
    load_floating_text, load_weight_decoded_fp8_text, plan_weight_decoded_fp8_text,
)
from venturi.models.flash_next_moe import TensorExpertStore
from venturi.models.flash_next_request import FlashNextRequest


def fixture(path, *, execution_dtype=torch.bfloat16):
    path.mkdir(parents=True, exist_ok=True)
    cp, baseline, _ = floating_fixture(path/'original-floating', dtype=torch.bfloat16)
    baseline.to(dtype=execution_dtype)
    weights = load_file(str(cp.files[0]))
    root = 'model.language_model'
    for i, layer in enumerate(baseline.layers):
        prefix = f'{root}.layers.{i}.mlp.experts'
        gu, down = weights.pop(prefix+'.gate_up_proj'), weights.pop(prefix+'.down_proj')
        decoded = {}
        for expert in range(4):
            matrices = []
            for name, value in [('gate_proj', gu[expert, :6]), ('up_proj', gu[expert, 6:]), ('down_proj', down[expert])]:
                scale = torch.tensor([[max(float(value.float().abs().max())/448., 1e-6)]], dtype=torch.bfloat16)
                codes = (value.float()/scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
                weights[f'{prefix}.{expert}.{name}.weight'] = codes
                weights[f'{prefix}.{expert}.{name}.weight_scale_inv'] = scale
                # Independent fully expanded decode; never calls production
                # block decoder/store. Toy matrices each occupy one block.
                matrices.append((codes.float()*scale.float()).to(execution_dtype))
            decoded[expert] = (torch.cat(matrices[:2]), matrices[2])
        layer.mlp.expert_store = TensorExpertStore(layer.mlp.geometry, decoded)
    prefix = root+'.layers.1.ple.ple_embedding.ngram_embedding'
    global_scale = torch.tensor([.0015], dtype=torch.bfloat16)
    table = []
    for part in range(8):
        name = f'{prefix}.shard_{part}.weight'
        codes = (weights[name].float()/global_scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
        weights[name] = codes
        table.append(codes.bfloat16()*global_scale)
    weights[prefix+'.weight_scale'] = global_scale
    raw = dict(cp.config)
    raw['quantization_config'] = dict(quant_method='fp8', activation_scheme='dynamic', weight_block_size=[128, 128],
        weight_per_tensor=False, act_per_tensor=False, modules_to_convert=['ple.ple_embedding.ngram_embedding'])
    save_file(weights, path/'model.safetensors')
    (path/'config.json').write_text(json.dumps(raw))
    return inspect_checkpoint(path), baseline.eval(), torch.cat(table)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_explicit_fp8_bridge_matches_independently_decoded_floating_text(tmp_path, dtype):
    cp, expected, table = fixture(tmp_path, execution_dtype=dtype)
    loaded = load_weight_decoded_fp8_text(cp, **OPTIONS)
    loaded.core.to(dtype=dtype)
    assert loaded.report['weight_decoding_reference_supported'] is True
    assert loaded.report['activation_quantization_supported'] is False
    assert loaded.report['fp8_gemm_supported'] is False
    with sdpa_kernel(SDPBackend.MATH):
        a, b = FlashNextRequest(loaded.core, loaded.tables[1]), FlashNextRequest(expected, loaded.tables[1])
        for ids, bits, accepted in (([2, 31, 5], None, 3), ([3, 4, 5], [True, False, True], 2), ([8], None, 1)):
            x, y = a.preview(ids, mask=bits), b.preview(ids, mask=bits)
            assert torch.equal(x.logits, y.logits)
            a.commit(x, accepted)
            b.commit(y, accepted)
            assert_states_equal(a.state, b.state)
            assert torch.equal(a.last_logits, b.last_logits)
    from venturi.models.flash_next_ngram import NGramRequest
    reader = loaded.tables[1]
    plan = NGramRequest(reader.config, reader.multipliers).preview([2, 31, 5])
    rows = reader.lookup(plan)
    assert torch.equal(rows, table[torch.tensor(plan.rows)].flatten(-2))
    with pytest.raises(ValueError, match='separate codec'):
        load_floating_text(cp, **OPTIONS)


def test_fp8_bridge_preflight_scale_static_dtype_and_explicit_budgets(tmp_path):
    cp, _, _ = fixture(tmp_path)
    for options in (dict(OPTIONS, max_core_bytes=1), dict(OPTIONS, max_expert_bytes=1)):
        with pytest.raises(ValueError, match='budget'):
            plan_weight_decoded_fp8_text(cp, **options)
    mapping = dict(cp.tensors)
    mapping.pop('model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale')
    with pytest.raises(ValueError, match='global multiplier'):
        plan_weight_decoded_fp8_text(replace(cp, tensors=MappingProxyType(mapping)), **OPTIONS)
    mapping = {n: replace(s, dtype='F32') if s.dtype == 'BF16' and '.experts.' not in n and
               '.ple.ple_embedding.' not in n else s for n, s in cp.tensors.items()}
    with pytest.raises(ValueError, match='BF16 static'):
        plan_weight_decoded_fp8_text(replace(cp, tensors=MappingProxyType(mapping)), **OPTIONS)
