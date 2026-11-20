"""Qualify standalone Ops, then compare layer composition bit-for-bit.

This diagnostic deliberately uses separate temporaries, not DecoderWorkspace or
DecoderStep._execute, so incorrect arena lifetimes or ordering cannot self-verify.
"""

import torch

from attention_reference import attention_reference, rope_reference
from block_checks import qualified_native_composition
from block_reference import decode
from decoder_reference import norm_reference
from venturi.ops import Q4RowSplitWeight, Q8RowSplitWeight


def qualify(actual, reference, *, atol=2e-5):
    reference = reference.cpu().double()
    error = (actual.cpu().double() - reference.to(actual.dtype).double()).abs()
    bound = 2 * torch.finfo(actual.dtype).eps * reference.abs() + atol
    result = dict(
        passed=bool((error <= bound).all()),
        max_abs_error=error.max().item(),
        failed_elements=int((~(error <= bound)).sum()),
        criterion=f"abs(actual-rounded_FP64) <= 2*eps*abs(FP64) + {atol}",
    )
    assert result["passed"], result
    return result


def qualified_decoder_composition(step, weights, x, cos, sin, *, prefix=None):
    plan, dims = step.layer, step.layer.resident.dimensions
    t, h = x.shape
    q = torch.empty(
        (t, dims.query_heads, dims.head_dim), dtype=x.dtype, device=x.device
    )
    k = torch.empty((t, dims.kv_heads, dims.head_dim), dtype=x.dtype, device=x.device)
    v = torch.empty_like(k)
    norm, attended, projected = (
        torch.empty_like(x),
        torch.empty_like(q),
        torch.empty_like(x),
    )
    scratch = (
        torch.empty(plan.linear_workspace_bytes, dtype=torch.uint8, device=x.device)
        if plan.linear_workspace_bytes
        else None
    )
    checks = {}
    plan.norm.run(x, out=norm)
    # Keep the FP64 norm before activation cast for the error bound.
    checks["attention_norm"] = qualify(
        norm,
        norm_reference(x.double(), weights.attention_norm.double(), plan.norm.epsilon),
    )
    for name, operation, weight, target in (
        ("query", plan.query, weights.query, q),
        ("key", plan.key, weights.key, k),
        ("value", plan.value, weights.value, v),
    ):
        operation.run(
            norm,
            out=target.view(t, -1),
            workspace=scratch if operation.workspace_bytes else None,
        )
        checks[name] = qualify(
            target.view(t, -1),
            norm.cpu().double() @ decode(weight).T,
            atol=2e-4
            if isinstance(weight, (Q4RowSplitWeight, Q8RowSplitWeight))
            else 2e-5,
        )
    for name, operation, target in (
        ("rope_q", plan.rope_q, q),
        ("rope_k", plan.rope_k, k),
    ):
        expected = rope_reference(target, cos, sin, dims.rotary_dim, dims.rope_layout)
        operation.run(target, cos, sin, out=target)
        checks[name] = qualify(target, expected)
    keys = torch.cat((prefix[0], k)) if prefix is not None else k
    values = torch.cat((prefix[1], v)) if prefix is not None else v
    expected = attention_reference(q, keys, values, step.attention.scale)
    step.attention.run(q, keys, values, out=attended)
    checks["attention"] = qualify(attended, expected)
    flat = attended.view(t, dims.query_width)
    plan.output.run(
        flat, out=projected, workspace=scratch if plan.output.workspace_bytes else None
    )
    checks["output_projection"] = qualify(
        projected,
        flat.cpu().double() @ decode(weights.output).T,
        atol=2e-4
        if isinstance(weights.output, (Q4RowSplitWeight, Q8RowSplitWeight))
        else 2e-5,
    )
    residual = torch.empty_like(x)
    plan.add.run(x, projected, out=residual)
    checks["attention_residual"] = qualify(
        residual, x.cpu().double() + projected.cpu().double()
    )
    out = qualified_native_composition(plan.mlp, weights.mlp, residual)
    checks["mlp"] = dict(
        passed=True,
        independently_qualified_ops=6,
        criterion="unchanged fixed per-Op FP64 bounds; Q4 absolute floor 2e-4, others 2e-5",
    )
    return out, (keys, values), checks
