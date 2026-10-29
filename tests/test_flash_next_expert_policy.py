"""Setup-only explicit policy bindings, no silent backend/default/device changes."""

from dataclasses import replace
from types import MappingProxyType

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next_fp8_loader import fixture
from test_flash_next_loader import OPTIONS
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_request import FlashNextRequest


def test_cpu_policy_cache_preserves_predictions_and_device(tmp_path):
    cp, _, _ = fixture(tmp_path)
    bundle = load_weight_decoded_fp8_text(cp, **OPTIONS)
    with sdpa_kernel(SDPBackend.MATH):
        old = FlashNextRequest(bundle.core, bundle.tables[1]).preview([2, 31, 5]).logits
        policy = bind_fp8_expert_policy(bundle.core, cp, backend='cpu-vectorized', retained_bytes_per_layer=(1024, 1024))
        req = FlashNextRequest(bundle.core, bundle.tables[1])
        new = req.preview([2, 31, 5]).logits
    assert torch.equal(old, new) and policy['execution_device'] == 'cpu'
    assert not policy['native_full_model'] and not policy['automatic_oom_fallback']


def test_all_store_admission_precedes_any_rebinding(tmp_path):
    cp, _, _ = fixture(tmp_path)
    bundle = load_weight_decoded_fp8_text(cp, **OPTIONS)
    before = [layer.mlp.expert_store for layer in bundle.core.layers]
    mapping = dict(cp.tensors)
    mapping.pop('model.language_model.layers.1.mlp.experts.3.down_proj.weight')
    with pytest.raises(ValueError, match='required'):
        bind_fp8_expert_policy(bundle.core, replace(cp, tensors=MappingProxyType(mapping)),
            backend='cpu-vectorized', retained_bytes_per_layer=(1024, 1024))
    assert [layer.mlp.expert_store for layer in bundle.core.layers] == before
    for backend, budgets in [('auto', (0, 0)), ('cpu-vectorized', (0,)), ('cpu-tiled', (True, 0)),
                             ('cuda-byte-decode', (0, 0))]:
        with pytest.raises(ValueError):
            bind_fp8_expert_policy(bundle.core, cp, backend=backend, retained_bytes_per_layer=budgets)
    assert [layer.mlp.expert_store for layer in bundle.core.layers] == before
