"""Reproduce a whole-chain FP16 cancellation failure and preserve actual operands."""

import json
import hashlib
import os
from pathlib import Path
import torch
from block_checks import qualified_native_composition
from block_reference import gated_mlp
from test_gated_mlp import make_weights
from venturi.ops import CudaBackend

torch.set_num_threads(4)
weights = make_weights(40, 136, torch.float16)
backend = CudaBackend(os.environ["VENTURI_BLOCKS_LIBRARY"])
plan = weights.prepare(backend, tokens=7, dtype=torch.float16, device="cuda:0")
ws, out = plan.empty_workspace(), torch.empty(7, 40, device="cuda", dtype=torch.float16)
for seed in range(1000):
    xcpu = torch.randn(
        7, 40, dtype=torch.float16, generator=torch.Generator().manual_seed(seed)
    )
    x = xcpu.cuda()
    expected = gated_mlp(weights, xcpu, 1e-6)
    plan.run(x, out=out, workspace=ws)
    actual = out.cpu()
    error = (actual.double() - expected.double()).abs()
    bound = 4 * torch.finfo(x.dtype).eps * expected.double().abs() + 3e-5
    failures = (error > bound).nonzero()
    if len(failures):
        payload = dict(
            seed=seed,
            shape=[7, 40, 136],
            dtype="float16",
            x=xcpu.tolist(),
            actual=actual.tolist(),
            composite_fp64=expected.tolist(),
            failure_indices=failures.tolist(),
            criterion="4*eps*abs(composite)+3e-5",
            note="Composite diagnostic, not proof of a per-Op error. Qualify every Op on its actual represented inputs separately.",
        )
        qualified = qualified_native_composition(plan, weights, x)
        assert torch.equal(qualified.cpu(), actual)
        payload.update(
            all_six_ops_passed_independent_fp64=True,
            per_op_criterion="abs(actual-rounded_FP64) <= 2*eps*abs(FP64) + 2e-5",
            composition_bit_identical=True,
            represented_weight_seed=268,
            native_library_sha256=hashlib.sha256(
                Path(backend.library).read_bytes()
            ).hexdigest(),
            torch_version=torch.__version__,
        )
        Path("composite-fp16-counterexample.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        print("counterexample seed", seed, "indices", failures.tolist(), flush=True)
        break
else:
    raise RuntimeError("No composite counterexample in bounded search")
