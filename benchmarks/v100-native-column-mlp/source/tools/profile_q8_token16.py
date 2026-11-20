"""One actual T16 fused MLP: eager diagnostic calls, not model/serving throughput."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter
import torch
from venturi.models import QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, Q8RowSplitWeight
from venturi.ops.q8_fused import prepare_q8_fused


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    role = next(
        s.binding for s in weights.plan.bindings if s.binding.endswith("/mlp/gate")
    )
    prefix = role[: -len("mlp/gate")]
    sources = []
    for suffix in ["mlp/gate", "mlp/up"]:
        w = weights.read_linear(prefix + suffix)
        half = (w.codes.shape[0] + 1) // 2
        sources.append(
            Q8RowSplitWeight(
                w.codes[:half].clone(), w.scales[:half].clone(), w.in_features
            )
        )
    b = CudaBackend(a.library)
    resident = [
        b.upload_linear(w, dtype=torch.float16, device="cuda:0") for w in sources
    ]
    plan = prepare_q8_fused(resident, tokens=16, residual_compensation=True)
    x = (
        torch.randn(
            16, sources[0].in_features, generator=torch.Generator().manual_seed(1731)
        )
        .half()
        .cuda()
    )
    work = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    outputs = [
        torch.empty(16, w.codes.shape[0], dtype=torch.float16, device="cuda:0")
        for w in sources
    ]

    def call():
        plan.run(x, outputs=outputs, workspace=work)

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    before = [o.clone() for o in outputs]
    start, used = perf_counter(), cpu()
    with torch.cuda.nvtx.range("native_unsplit"):
        for _ in range(9):
            call()
        torch.cuda.synchronize()
    wall, used = perf_counter() - start, cpu() - used
    assert all(
        torch.equal(x.view(torch.uint8), y.view(torch.uint8))
        for x, y in zip(before, outputs)
    )
    Path(a.output).write_text(
        json.dumps(
            dict(
                scope=__doc__,
                profiled=True,
                native_call_ranges=1,
                logical_requests=None,
                eager_projection_calls=9,
                tokens=16,
                k=sources[0].in_features,
                ns=[w.codes.shape[0] for w in sources],
                projection_profile="q8-residual",
                wall_seconds=wall,
                process_cpu_core_seconds=used,
                library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
                output_sha256=[
                    hashlib.sha256(o.cpu().numpy().tobytes()).hexdigest()
                    for o in outputs
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
