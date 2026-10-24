"""Per-Op qualification on actual inputs plus exact composition equivalence.

The numerical oracle is independent FP64 math, not a different GPU backend.
Whole-chain FP64 rounding is diagnostic: an early output-cast boundary may
propagate through residual cancellation despite every Op meeting its bound.
"""

import torch

from block_reference import decode
from venturi.ops import Q4RowSplitWeight, Q8RowSplitWeight


def qualified_native_composition(plan, weights, x):
    ws = plan.empty_workspace()
    out = torch.empty_like(x)

    def check(actual, fp64, atol=2e-5):
        rounded = fp64.to(actual.dtype).double()
        error = (actual.cpu().double() - rounded).abs()
        bound = 2 * torch.finfo(actual.dtype).eps * fp64.abs() + atol
        assert bool((error <= bound).all()), dict(
            max_abs_error=float(error.max()),
            failures=(error > bound).nonzero().tolist(),
        )

    source = x.cpu().double()
    plan.norm.run(x, out=ws.hidden)
    check(
        ws.hidden,
        source
        * (source.square().mean(-1, keepdim=True) + plan.norm.epsilon).rsqrt()
        * weights.norm.cpu().double(),
    )
    normalized = ws.hidden.cpu().double()
    for operation, weight, target in (
        (plan.gate, weights.gate, ws.gate),
        (plan.up, weights.up, ws.up),
    ):
        operation.run(
            ws.hidden,
            out=target,
            workspace=ws.linear if operation.workspace_bytes else None,
        )
        check(
            target,
            normalized @ decode(weight).T,
            atol=2e-4
            if isinstance(weight, (Q4RowSplitWeight, Q8RowSplitWeight))
            else 2e-5,
        )
    gate, up = ws.gate.cpu().double(), ws.up.cpu().double()
    plan.swiglu.run(ws.gate, ws.up, out=ws.gate)
    check(ws.gate, gate * torch.sigmoid(gate) * up)
    activated = ws.gate.cpu().double()
    plan.down.run(
        ws.gate,
        out=ws.hidden,
        workspace=ws.linear if plan.down.workspace_bytes else None,
    )
    check(
        ws.hidden,
        activated @ decode(weights.down).T,
        atol=2e-4
        if isinstance(weights.down, (Q4RowSplitWeight, Q8RowSplitWeight))
        else 2e-5,
    )
    plan.add.run(x, ws.hidden, out=out)
    check(out, source + ws.hidden.cpu().double())
    return out
