"""Floating router/expert/shared-gate contracts; not a complete model test."""

from dataclasses import replace
import json

import pytest
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_moe import (
    FlashNextMoE, FusedFloatingExpertStore, MoEGeometry,
    TensorExpertStore, routed_experts,
)


def geometry(normalize=True):
    return MoEGeometry(8, 4, 2, 6, 5, normalize)


def make_module(dtype=torch.float32, normalize=True):
    torch.manual_seed(420)
    g = geometry(normalize)
    bank = TensorExpertStore(g, {e: (torch.randn(12, 8, dtype=dtype) * 0.2,
                                    torch.randn(8, 6, dtype=dtype) * 0.2) for e in range(4)})
    module = FlashNextMoE(g, bank).to(dtype)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    return module.eval()


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_router_float_softmax_and_optional_renormalization(dtype, normalize):
    module = make_module(dtype, normalize)
    x = torch.randn(2, 3, 8, dtype=dtype)
    logits, scores, indices = module.gate(x)
    expected_logits = F.linear(x.reshape(-1, 8), module.gate.weight)
    probabilities = torch.softmax(expected_logits.float(), -1)
    expected_scores, expected_ids = torch.topk(probabilities, 2, -1)
    if normalize:
        expected_scores = expected_scores / expected_scores.sum(-1, keepdim=True)
    assert torch.equal(logits, expected_logits)
    assert torch.equal(indices, expected_ids)
    assert torch.equal(scores, expected_scores.to(dtype))
    if not normalize:
        assert torch.all(scores.float().sum(-1) < 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8,), (2, 3, 8), (0, 1, 8)])
def test_selected_expert_and_shared_gate_against_tokenwise_formula(dtype, shape):
    module = make_module(dtype)
    x = torch.randn(shape, dtype=dtype)
    before = x.clone()
    flat = x.reshape(-1, 8)
    _, scores, indices = module.gate(flat)
    expected = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for expert in sorted(indices[token].tolist()):
            position = indices[token].tolist().index(expert)
            gate_up, down = module.expert_store.read(expert, device="cpu", dtype=dtype)
            gate, up = F.linear(flat[token:token + 1], gate_up).chunk(2, -1)
            value = F.linear(F.silu(gate) * up, down) * scores[token, position]
            expected[token] += value[0]
    shared = module.shared_expert
    shared_output = shared["down_proj"](F.silu(shared["gate_proj"](flat)) * shared["up_proj"](flat))
    expected = (expected + torch.sigmoid(module.shared_expert_gate(flat)) * shared_output).reshape(shape)
    actual = module(x)
    torch.testing.assert_close(actual, expected, atol=2e-6 if dtype == torch.float32 else 0.004,
                               rtol=2e-5 if dtype == torch.float32 else 0.035)
    assert torch.equal(x, before)


def test_ties_extreme_logits_noncontiguous_and_different_batch_rows():
    module = make_module()
    x = torch.randn(3, 2, 8).transpose(0, 1)
    assert torch.equal(module(x), module(x.contiguous()))
    with torch.no_grad():
        module.gate.weight.zero_()
    _, scores, ids = module.gate(x)
    assert torch.equal(ids, torch.topk(torch.ones(6, 4) / 4, 2, -1).indices)
    assert torch.equal(scores, torch.full((6, 2), 0.5))
    with torch.no_grad():
        module.gate.weight[0].fill_(100)
    _, scores, _ = module.gate(torch.full((1, 8), 100.))
    assert torch.isfinite(scores).all()


def test_missing_selected_expert_cannot_be_silently_pruned():
    module = make_module()
    module.expert_store.experts.pop(2)
    with pytest.raises(ValueError, match="cannot prune"):
        routed_experts(torch.ones(1, 8), torch.tensor([[1, 2]]), torch.full((1, 2), 0.5), module.expert_store)
    for ids in ([[0, -1]], [[0, 4]]):
        with pytest.raises(ValueError, match="real expert ids"):
            routed_experts(torch.ones(1, 8), torch.tensor(ids), torch.ones(1, 2), module.expert_store)


@pytest.mark.parametrize("field,value", [("hidden_size", True), ("top_k", 5), ("num_experts", 0),
                                        ("normalize_topk", None)])
def test_invalid_geometry(field, value):
    with pytest.raises(ValueError):
        replace(geometry(), **{field: value})


def test_pinned_config_default_and_activation_rejection():
    c = dict(model_type="qwen4_exp", text_config=dict(model_type="qwen4_exp_text", hidden_size=8,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=6, shared_expert_intermediate_size=5))
    assert MoEGeometry.from_dict(c).normalize_topk is True
    c["text_config"]["norm_topk_prob"] = False
    assert MoEGeometry.from_dict(c).normalize_topk is False
    c["text_config"]["norm_topk_prob"] = None
    with pytest.raises(ValueError):
        MoEGeometry.from_dict(c)
    c["text_config"]["hidden_act"] = "relu"
    with pytest.raises(ValueError, match="SiLU"):
        MoEGeometry.from_dict(c)
    with pytest.raises(ValueError):
        MoEGeometry.from_dict({"model_type": "qwen3_5"})


def test_expert_bank_and_routing_shape_errors():
    module = make_module()
    with pytest.raises(ValueError, match="geometry"):
        TensorExpertStore(geometry(), {0: (torch.zeros(12, 8), torch.zeros(8, 5))})
    with pytest.raises(ValueError, match="hidden size"):
        module(torch.zeros(2, 7))
    with pytest.raises(ValueError, match="dtype"):
        module.gate(torch.zeros(2, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="geometry"):
        routed_experts(torch.zeros(2, 8), torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 1), module.expert_store)
    with pytest.raises(ValueError, match="dtype"):
        routed_experts(torch.zeros(2, 8), torch.zeros(2, 2), torch.zeros(2, 2), module.expert_store)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_checkpoint_slice_and_core_binding(tmp_path, dtype, monkeypatch):
    module = make_module(dtype)
    prefix = "model.language_model.layers.0.mlp"
    weights = {f"{prefix}.{k}": t.detach().clone() for k, t in module.state_dict().items()}
    weights[prefix + ".experts.gate_up_proj"] = torch.stack([module.expert_store.experts[i][0] for i in range(4)])
    weights[prefix + ".experts.down_proj"] = torch.stack([module.expert_store.experts[i][1] for i in range(4)])
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    checkpoint = inspect_checkpoint(tmp_path)
    bound = FlashNextMoE.from_checkpoint(checkpoint, geometry(), prefix)
    x = torch.randn(2, 3, 8, dtype=dtype)
    assert torch.equal(bound(x), module(x))
    # Prove the expert path requests slices, never the complete 3D bank.
    import venturi.models.flash_next_moe as impl
    real_open = impl.safe_open

    class SliceOnly:
        def __init__(self, *args, **kwargs):
            self.context = real_open(*args, **kwargs)

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_tensor(self, name):
            raise AssertionError("Full expert bank must not be materialized")

        def get_slice(self, name):
            return self.reader.get_slice(name)

    monkeypatch.setattr(impl, "safe_open", SliceOnly)
    gate_up, down = bound.expert_store.read(2, device="cpu", dtype=dtype)
    assert torch.equal(gate_up, weights[prefix + ".experts.gate_up_proj"][2])
    assert torch.equal(down, weights[prefix + ".experts.down_proj"][2])
    (tmp_path / "model.safetensors").unlink()
    assert torch.isfinite(gate_up).all() and torch.isfinite(down).all()
    with pytest.raises(ValueError, match="budget"):
        FusedFloatingExpertStore(checkpoint, geometry(), prefix + ".experts", max_expert_bytes=1)
    with pytest.raises(ValueError, match="core budget"):
        FlashNextMoE.from_checkpoint(checkpoint, geometry(), prefix, max_core_bytes=1)
    broken = replace(checkpoint, tensors={k: v for k, v in checkpoint.tensors.items()
                                         if not k.endswith("experts.down_proj")})
    with pytest.raises(ValueError, match="FP8 needs its own codec"):
        FusedFloatingExpertStore(broken, geometry(), prefix + ".experts")
