"""Separate standalone projection attribution; never a serving TPS benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from venturi.models import QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a
from venturi.ops.q8_shared import prepare_q8_tiled_shared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--baseline-library", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    spec = next(
        s
        for s in weights.plan.bindings
        if s.shape == (17408, 5120) and s.format == "q8_g32_fp16"
    )
    full = weights.read_linear(spec.binding)
    source = Q8RowSplitWeight(
        full.codes[:8704].clone(), full.scales[:8704].clone(), 5120
    )
    backend, control = CudaBackend(a.library), CudaBackend(a.baseline_library)
    resident, old = [
        b.upload_linear(source, dtype=torch.float16, device="cuda:0")
        for b in (backend, control)
    ]
    scope = dict(
        profiled=True,
        scope=__doc__,
        shape=[8704, 5120],
        tokens=[4, 8, 32],
        calls_per_range=16,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        baseline_library_sha256=hashlib.sha256(
            Path(a.baseline_library).read_bytes()
        ).hexdigest(),
        ranges=[],
    )
    for tokens in scope["tokens"]:
        x = (
            torch.randn(
                tokens, 5120, generator=torch.Generator().manual_seed(719 + tokens)
            )
            .half()
            .cuda()
        )
        plans = [
            prepare_q8_dp4a(r, tokens=tokens, residual_compensation=True)
            for r in (old, resident)
        ] + [
            prepare_q8_tiled_shared(
                resident,
                tokens=tokens,
                rows=rows,
                token_tile=tile,
                residual_compensation=True,
            )
            for tile in (4, 8)
            for rows in (4, 8, 16)
        ]
        outputs = [
            torch.empty(tokens, 8704, dtype=torch.float16, device="cuda:0")
            for _ in plans
        ]
        scratch = [
            torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
            for p in plans
        ]
        labels = ["control-residual", "candidate-residual"] + [
            f"ktile{t}-rows{r}" for t in (4, 8) for r in (4, 8, 16)
        ]

        def call(i):
            plans[i].run(x, out=outputs[i], workspace=scratch[i])

        for i in range(len(plans)):
            call(i)
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range("native_unsplit"):
            for i, label in enumerate(labels):
                name = f"projection_{label}_t{tokens}"
                with torch.cuda.nvtx.range(name):
                    for _ in range(16):
                        call(i)
                torch.cuda.synchronize()
                scope["ranges"].append(
                    dict(name=name, profile=label, tokens=tokens, calls=16)
                )
    Path(a.output).write_text(json.dumps(scope, indent=2) + "\n")


if __name__ == "__main__":
    main()
