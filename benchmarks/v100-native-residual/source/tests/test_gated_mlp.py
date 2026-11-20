"""Shared mathematics, binding ownership, workspace and inference contracts."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from block_reference import gated_mlp
from venturi.models.gated_mlp import GatedMLPWeights, GatedMLPWorkspace
from venturi.ops import DenseWeight, Q4RowSplitWeight, TorchBackend


def make_weights(h, intermediate, dtype=torch.float64, q4=False):
    gen = torch.Generator().manual_seed(92 + h + intermediate)

    def weight(n, k):
        if not q4:
            return DenseWeight(torch.randn(n, k, generator=gen, dtype=dtype) / k**0.5)
        codes = torch.randint(-8, 8, (n, k), generator=gen, dtype=torch.int8)
        scales = torch.full((n, (k + 63) // 64), 0.02 / k**0.5, dtype=torch.float16)
        return Q4RowSplitWeight.from_codes(codes, scales)

    return GatedMLPWeights(
        torch.rand(h, generator=gen, dtype=dtype) + 0.5,
        weight(intermediate, h),
        weight(intermediate, h),
        weight(h, intermediate),
    )


@pytest.mark.parametrize("h,intermediate", [(32, 80), (40, 136), (65, 97)])
@pytest.mark.parametrize("tokens", [1, 7, 17])
@pytest.mark.parametrize("q4", [False, True])
def test_shared_block_against_independent_staged_fp64(h, intermediate, tokens, q4):
    weights = make_weights(h, intermediate, q4=q4)
    plan = weights.prepare(
        TorchBackend(), tokens=tokens, dtype=torch.float64, device="cpu"
    )
    workspace = plan.empty_workspace()
    x = torch.randn(tokens, h, dtype=torch.float64)
    out = torch.empty_like(x)
    expected = gated_mlp(weights, x, 1e-6)
    assert plan.run(x, out=out, workspace=workspace) is out
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
    assert workspace.nbytes == 8 * tokens * (h + 2 * intermediate)
    x.zero_()
    plan.run(x, out=out, workspace=workspace)
    assert torch.equal(out, torch.zeros_like(out))


def test_preparation_owns_weights_and_workspaces_are_independent():
    weights = make_weights(16, 32)
    plan = weights.prepare(TorchBackend(), tokens=2, dtype=torch.float64, device="cpu")
    a, b = plan.empty_workspace(), plan.empty_workspace()
    x = torch.randn(2, 16, dtype=torch.float64)
    out = torch.empty_like(x)
    expected = gated_mlp(weights, x, 1e-6)
    weights.norm.zero_()
    for w in (weights.gate, weights.up, weights.down):
        w.data.zero_()
    plan.run(x, out=out, workspace=a)
    torch.testing.assert_close(out, expected)
    a.gate.fill_(float("nan"))
    plan.run(x, out=out, workspace=b)
    torch.testing.assert_close(out, expected)
    with pytest.raises(FrozenInstanceError):
        plan.gate.tokens = 3


def test_workspace_shape_alias_and_autograd_rejected_before_execution():
    plan = make_weights(16, 32).prepare(
        TorchBackend(), tokens=2, dtype=torch.float64, device="cpu"
    )
    x = torch.randn(2, 16, dtype=torch.float64)
    out = torch.full_like(x, 123)
    ws = plan.empty_workspace()
    bad = GatedMLPWorkspace(ws.hidden, ws.gate, ws.gate, None)
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=out, workspace=bad)
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=x, workspace=ws)
    with pytest.raises(ValueError, match="shape"):
        plan.run(x.float(), out=out, workspace=ws)
    with pytest.raises(ValueError, match="inference"):
        plan.run(x.requires_grad_(), out=out, workspace=ws)
    assert torch.equal(out, torch.full_like(out, 123))


def test_binding_rejects_incompatible_projection_shapes():
    weights = make_weights(16, 32)
    with pytest.raises(ValueError, match="gate and up"):
        GatedMLPWeights(
            weights.norm, weights.gate, DenseWeight(torch.zeros(32, 17)), weights.down
        )
    with pytest.raises(ValueError, match="down"):
        GatedMLPWeights(
            weights.norm, weights.gate, weights.up, DenseWeight(torch.zeros(16, 33))
        )
    with pytest.raises(ValueError, match="norm"):
        GatedMLPWeights(torch.ones(15), weights.gate, weights.up, weights.down)
    with pytest.raises(TypeError, match="descriptors"):
        GatedMLPWeights(weights.norm, weights.gate.data, weights.up, weights.down)


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf"), True, 1e-50])
def test_invalid_epsilon_rejected(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        make_weights(8, 16).prepare(
            TorchBackend(), tokens=1, dtype=torch.float64, device="cpu", epsilon=epsilon
        )


@pytest.mark.parametrize(
    "data",
    [
        torch.zeros(2, 3, dtype=torch.int8),
        torch.zeros(2, 3).T,
        torch.zeros(0, 3),
        torch.zeros(3),
    ],
)
def test_dense_weight_storage_validation(data):
    with pytest.raises(ValueError):
        DenseWeight(data)


def test_q4_reference_does_not_round_stored_weights_to_bf16():
    from block_reference import decode

    weights = make_weights(65, 97, torch.bfloat16, q4=True)
    plan = TorchBackend().prepare_linear(
        weights.gate, tokens=7, dtype=torch.bfloat16, device="cpu"
    )
    generator = torch.Generator().manual_seed(337)
    x = torch.randn(7, 65, generator=generator).bfloat16()
    out = torch.empty(7, 97, dtype=torch.bfloat16)
    plan.run(x, out=out)
    expected = (x.double() @ decode(weights.gate).T).bfloat16()
    assert torch.equal(out, expected)
    assert plan.storage_tensors[0].dtype == torch.float32
