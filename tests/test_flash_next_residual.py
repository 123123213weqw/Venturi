"""Independent small floating fixtures for gated residual read/inject/mixer."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_residual import FlashNextGatedResidual, GroupedZeroCenteredRMSNorm


def make_module(dtype, combine=True):
    torch.manual_seed(913)
    module = FlashNextGatedResidual(8, 4, 5, use_combine=combine).to(dtype)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("shape", [(32,), (2, 3, 32), (0, 32)])
def test_grouped_read_inject_and_final_mixer(dtype, combine, shape):
    module = make_module(dtype, combine)
    x = torch.randn(shape, dtype=dtype)
    # Normalize each residual stream independently; include zero-centered weights
    # before the BF16 cast. Unequal stream scales distinguish global RMS.
    x = x * torch.tensor([0.01, 1, 7, 23], dtype=dtype).repeat_interleave(8)
    before = x.clone()
    chunks = []
    for stream in range(4):
        chunk = x[..., stream * 8:(stream + 1) * 8].float()
        normalized = chunk / torch.sqrt(chunk.square().mean(-1, keepdim=True) + 1e-6)
        weight = module.hc_norm.weight[stream * 8:(stream + 1) * 8].float()
        chunks.append((normalized * (1 + weight)).to(dtype))
    expected_norm = torch.cat(chunks, -1)
    # Independent division/sqrt vs multiplication/rsqrt may differ in FP32.
    torch.testing.assert_close(module.hc_norm(x), expected_norm, atol=2e-6 if dtype == torch.float32 else 0.03125,
                               rtol=2e-6 if dtype == torch.float32 else 0.016)
    normalized = module.hc_norm(x)
    low = F.silu(F.linear(normalized, module.input_mix_weight_down.weight) / 4)
    gates = torch.sigmoid(F.linear(low, module.input_mix_weight_up.weight)).reshape(*shape[:-1], 4, 8)
    expected_mix = (gates * normalized.reshape(*shape[:-1], 4, 8)).mean(-2)
    actual = module(x)
    mixed = actual[0] if combine else actual
    assert torch.equal(mixed.view(torch.int32 if dtype == torch.float32 else torch.int16),
                       expected_mix.view(torch.int32 if dtype == torch.float32 else torch.int16))
    if combine:
        _, residual, injection = actual
        assert residual is x
        expected_gate = 2 * torch.sigmoid(F.linear(normalized, module.block_inject_weight.weight) / 4)
        assert torch.equal(injection, expected_gate)
        output = torch.randn((*shape[:-1], 8), dtype=dtype)
        result = module.inject(output, residual, injection)
        expected = torch.cat([x[..., i * 8:(i + 1) * 8] + output * injection[..., i:i + 1]
                              for i in range(4)], -1)
        assert torch.equal(result, expected)
        assert result.data_ptr() != x.data_ptr() or x.numel() == 0
    assert torch.equal(before, x)


def test_grouped_norm_zero_center_and_noncontiguous_input():
    norm = GroupedZeroCenteredRMSNorm(2, 2)
    x = torch.tensor([[3., 4., 0., 0.], [0., 0., 3., 4.]])
    assert torch.allclose(norm(x), torch.tensor([[0.8485281, 1.1313708, 0, 0],
                                                [0, 0, 0.8485281, 1.1313708]]))
    with torch.no_grad():
        norm.weight.fill_(-1)
    assert torch.count_nonzero(norm(x)) == 0
    module = make_module(torch.float32)
    noncontiguous = torch.randn(3, 2, 32).transpose(0, 1)
    assert not noncontiguous.is_contiguous()
    assert torch.equal(module(noncontiguous)[0], module(noncontiguous.contiguous())[0])


def test_residual_validation():
    for args in [(0, 4), (8, True), (8, 4, float("nan"))]:
        with pytest.raises(ValueError):
            GroupedZeroCenteredRMSNorm(*args)
    with pytest.raises(ValueError):
        FlashNextGatedResidual(8, 4, 0)
    module = make_module(torch.float32)
    with pytest.raises(ValueError, match="flattened"):
        module(torch.zeros(2, 8))
    with pytest.raises(ValueError, match="dtype"):
        module(torch.zeros(2, 32, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="geometry"):
        module.inject(torch.zeros(2, 7), torch.zeros(2, 32), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="dtype"):
        module.inject(torch.zeros(2, 8, dtype=torch.bfloat16), torch.zeros(2, 32), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="does not inject"):
        make_module(torch.float32, False).inject(None, None, None)


@pytest.mark.parametrize("combine", [False, True])
def test_small_checkpoint_binding_owns_exact_weights(tmp_path, combine):
    module = make_module(torch.bfloat16, combine)
    prefix = "model.language_model.hyper_connection_mixer" if not combine else (
        "model.language_model.layers.0.attn_hyper_connection")
    save_file({f"{prefix}.{k}": v.detach().clone() for k, v in module.state_dict().items()},
              tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    checkpoint = inspect_checkpoint(tmp_path)
    config = SimpleNamespace(hc_count=4, hidden_size=8, hc_lowrank=5)
    loaded = FlashNextGatedResidual.from_checkpoint(checkpoint, config, prefix, eps=1e-6, use_combine=combine)
    assert not loaded.training
    for name, value in loaded.state_dict().items():
        assert torch.equal(value.view(torch.int16), module.state_dict()[name].view(torch.int16))
    (tmp_path / "model.safetensors").unlink()
    assert torch.isfinite(loaded(torch.zeros(2, 32, dtype=torch.bfloat16))[0]).all()
    missing = replace(checkpoint, tensors={k: v for k, v in checkpoint.tensors.items()
                                         if not k.endswith("hc_norm.weight")})
    with pytest.raises(ValueError, match="tensor required"):
        FlashNextGatedResidual.from_checkpoint(missing, config, prefix, eps=1e-6, use_combine=combine)
