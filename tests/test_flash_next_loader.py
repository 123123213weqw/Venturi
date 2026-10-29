"""Complete tiny floating checkpoint binding; not actual official model loading."""

from dataclasses import replace
import json
from types import MappingProxyType

import pytest
from safetensors.torch import save_file
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next_text import assert_states_equal, module, table_fixture
from test_flash_next import raw_config
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_loader import load_floating_text, plan_floating_text
from venturi.models.flash_next_request import FlashNextRequest


OPTIONS = dict(max_core_bytes=2**20, max_expert_bytes=2**20, max_sequence=128, attention_backend='sdpa')


def fixture(path, *, dtype=torch.float32):
    candidate = module(dtype)
    store, table = table_fixture(path, dtype=dtype)
    raw = raw_config()
    raw['text_config'].update(num_key_value_heads=2, num_attention_heads=4, head_dim=4,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=4, indexer_budget=4, indexer_compress_ratio=2,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, hidden_act='silu', output_gate_type='sigmoid', moe_intermediate_size=6,
        shared_expert_intermediate_size=6, ple_conv_kernel_size=4, rms_norm_eps=1e-6,
        mamba_ssm_dtype='float32', rope_parameters=dict(rope_type='default', rope_theta=10000000., partial_rotary_factor=.5))
    weights = {}
    root = 'model.language_model'
    for key, parameter in candidate.named_parameters():
        name = key if key.startswith('lm_head.') else root+'.'+key
        if key.startswith('layers.') and '.attention.' in key:
            i = int(key.split('.')[1])
            name = name.replace('.attention.', '.linear_attn.' if i == 0 else '.self_attn.', 1)
        weights[name] = parameter.detach().clone()
    for i, layer in enumerate(candidate.layers):
        for name, index in (('gate_up_proj', 0), ('down_proj', 1)):
            weights[f'{root}.layers.{i}.mlp.experts.{name}'] = torch.stack([
                layer.mlp.expert_store.experts[e][index] for e in range(4)]).to(dtype)
    for i, spec in enumerate(store.parts):
        rows = store.rows_per_part
        weights[spec.name] = table[i*rows:(i+1)*rows].clone()
    weights[root+'.layers.1.ple.ple_embedding.layer_multipliers'] = torch.tensor(store.multipliers)
    # Extra components are retained on disk, never bound by this text-only API.
    weights['model.visual.fixture'] = torch.ones(7, dtype=dtype)
    weights['mtp.fixture'] = torch.ones(9, dtype=dtype)
    save_file(weights, path/'model.safetensors')
    (path/'config.json').write_text(json.dumps(raw))
    return inspect_checkpoint(path), candidate, table


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_complete_floating_binding_and_transaction_are_exact(tmp_path, dtype):
    cp, candidate, _ = fixture(tmp_path, dtype=dtype)
    loaded = load_floating_text(cp, **OPTIONS)
    assert loaded.report['resident_parameter_bytes'] == sum(p.numel()*p.element_size() for p in candidate.parameters())
    assert loaded.report['file_backed_expert_bytes'] > 0 and loaded.report['file_backed_ple_bytes'] > 0
    assert loaded.report['native_generation_supported'] is False
    for key, parameter in loaded.core.named_parameters():
        assert torch.equal(parameter, dict(candidate.named_parameters())[key])
        assert parameter.device.type == 'cpu' and parameter.dtype == dtype
    a, b = FlashNextRequest(loaded.core, loaded.tables[1]), FlashNextRequest(candidate, loaded.tables[1])
    with sdpa_kernel(SDPBackend.MATH):
        for ids, bits, accepted in (([1, 31, 5], None, 3), ([2, 3, 4], [True, False, True], 2), ([6], None, 1)):
            p, q = a.preview(ids, mask=bits), b.preview(ids, mask=bits)
            assert torch.equal(p.logits, q.logits)
            a.commit(p, accepted)
            b.commit(q, accepted)
            assert_states_equal(a.state, b.state)
            assert a.history == b.history


def test_preflight_metadata_only_and_core_does_not_read_expert_or_table_payloads(tmp_path, monkeypatch):
    cp, _, _ = fixture(tmp_path)
    import venturi.models.flash_next_loader as loader
    original, reads = loader.safe_open, []
    class Reader:
        def __init__(self, *args, **kwargs):
            self.handle = original(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def get_slice(self, name):
            return self.handle.get_slice(name)
        def get_tensor(self, name):
            reads.append(name)
            return self.handle.get_tensor(name)
    monkeypatch.setattr(loader, 'safe_open', Reader)
    report = plan_floating_text(cp, **OPTIONS)
    assert not reads
    loaded = load_floating_text(cp, **OPTIONS)
    assert len(reads) == report['static_tensor_count']
    assert all('.experts.' not in n and '.ngram_embedding.' not in n for n in reads)
    assert loaded.core._inv_freq.device.type == 'cpu' and loaded.core._inv_freq.dtype == torch.float32
    with pytest.raises(TypeError):
        loaded.tables[1] = None


def test_preflight_all_required_weights_and_budgets_before_payload_reads(tmp_path, monkeypatch):
    cp, _, _ = fixture(tmp_path)
    import venturi.models.flash_next_loader as loader
    def forbidden(*args, **kwargs):
        raise AssertionError('payload access before successful metadata admission')
    monkeypatch.setattr(loader, 'safe_open', forbidden)
    for options in (dict(OPTIONS, max_core_bytes=1), dict(OPTIONS, max_expert_bytes=1),
                    dict(OPTIONS, max_sequence=True), dict(OPTIONS, attention_backend='auto')):
        with pytest.raises(ValueError):
            load_floating_text(cp, **options)
    mapping = dict(cp.tensors)
    mapping.pop('model.language_model.hyper_connection_mixer.hc_norm.weight')
    with pytest.raises(ValueError, match='Missing'):
        load_floating_text(replace(cp, tensors=MappingProxyType(mapping)), **OPTIONS)
    mapping = dict(cp.tensors)
    name = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight'
    mapping[name] = replace(mapping[name], dtype='I8')
    with pytest.raises(ValueError, match='floating PLE'):
        load_floating_text(replace(cp, tensors=MappingProxyType(mapping)), **OPTIONS)
    for key, value in [('rope_parameters', {'rope_type':'dynamic'}), ('mamba_ssm_dtype', 'bfloat16')]:
        raw = json.loads(json.dumps(dict(cp.config)))
        raw['text_config'][key] = value
        with pytest.raises(ValueError):
            load_floating_text(replace(cp, config=raw), **OPTIONS)
    raw = dict(cp.config, quantization_config={'quant_method':'fp8'})
    with pytest.raises(ValueError, match='separate codec'):
        load_floating_text(replace(cp, config=raw), **OPTIONS)
