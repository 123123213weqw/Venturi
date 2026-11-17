# SM70 Q5 Linear: bounded cuBLAS and explicit adaptive dispatch

This is a model-independent operator backend, not a new whole-model executor,
production service migration, or measured generation-TPS claim. Scope is V100
(SM70), `q5_g64_fp16` / `row_split_k128_v1`, contiguous FP16 X/Y and FP32
accumulation. No activation quantization or alternate model backend is selected.

## Numerical contract and fixes

Every weight is decoded as `FP16(signed_q5 * stored_FP16_scale)` before the dot.
Scalar, WMMA, the standalone Q5 reference and cuBLAS use this same boundary.
Different FP32 accumulation/reduction orders need not be bit-identical; tests use
an independent FP64 dot over the decoded FP16 values and FP16 output tolerances.
The previous scalar Q5 kernel retained FP32 scaling, unlike WMMA; that numerical
profile is intentionally superseded. Old whole-model trajectory evidence must
not be assumed to qualify this change.

The scalar pair loader preserves an odd final K element and handles half2-unaligned
token rows. WMMA still requires K divisible by 64; unsupported shapes cannot enter
that lane. Q6's existing bounded dequant+GEMM fallback is preserved instead of
being rejected by the new Q4/Q5-only registration table. ABI-17 stays additive;
the Q5 extension has its own version export. Matrix/Layer descriptor-size tests
now reflect the already-added per-matrix bits/group fields.

## Two execution lanes

* `cublas`: coalesced Q5 reconstruction into bounded caller-owned scratch, then
  `cublasGemmEx` with FP16 operands, FP32 accumulation and Tensor Cores permitted.
  Reconstruction occurs on **every** invocation and is included in timing.
  Chunks write into the final output using its leading dimension, with no output
  gather/copy kernel. The default total scratch limit is 64 MiB, including 4 MiB
  cuBLAS scratch. No persistent FP16 weight shadow is allocated.
* `adaptive`: choose scalar, WMMA or the same cuBLAS lane using an explicitly
  calibrated exact `(weight, M, workspace budget, library, GPU, precision policy)`
  record. No valid record means cuBLAS, not a guessed crossover. Calibration
  rejects numerically failing candidates and requires a configurable minimum
  GPU-stream-time advantage (3% by default) over cuBLAS; ties retain cuBLAS.

cuBLAS is allowed to select its normal algorithm; it is not deliberately kept on
a slow non-Tensor-Core algorithm. The legacy dense operator's pedantic arithmetic
policy is unchanged. A **separate diagnostic** measures cuBLAS against a resident
FP16 matrix containing the exact same decoded Q5 values, excluding reconstruction.
That is a stronger single-operator control, not evidence that an entire 27B FP16
model fits in VRAM.

## Public use

```python
import json
import torch
from venturi.ops import (CudaBackend, Q5RowSplitWeight, Q5Dispatcher,
                         tune_q5, export_q5_calibration, load_q5_calibration)

backend = CudaBackend("/absolute/path/to/prebuilt/libventuri_ops.so")
weight = Q5RowSplitWeight.from_dense(cpu_dense_weight)  # explicit offline quantization
resident = backend.upload_linear(weight, dtype=torch.float16, device="cuda:0")

# Setup only, on representative caller-supplied activations; never in capture/run.
for x in representative_inputs:
    tune_q5(resident, x, minimum_gain=0.03)
records = export_q5_calibration(resident)
# Save records explicitly if desired. A different weight, DSO or GPU UUID is rejected.
load_q5_calibration(resident, json.loads(json.dumps(records)))

baseline = Q5Dispatcher(resident, [1, 4, 16, 64, 256], route="cublas")
adaptive = Q5Dispatcher(resident, [1, 4, 16, 64, 256], route="adaptive")
workspace = torch.empty(adaptive.workspace_bytes, device="cuda:0", dtype=torch.uint8)
adaptive.run(x, out=caller_owned_output, workspace=workspace)
```

M means actual grouped token rows, not HTTP request concurrency or GPU busy.
Dispatch only looks up a prebound plan. Unknown M fails; there is no hidden timing,
compilation, padding, allocation, or fallback. Prepare after calibration: already
prepared plans and captured graphs keep their original route. Callers own outputs
and independent per-execution workspaces, and retain weights/plans/operands until
queued GPU work or graph replay completes. Close cuBLAS plans only after completion.
Public GatedMLP and standard decoder weight descriptors accept Q5; the dedicated
native-Qwen serving loop does not use this Python dispatcher. Its separate native
service integration and whole-model qualification are documented in
`qwen-q5-service-sm70.md`.

## Reproduction

Build only on the authorized remote Linux CUDA host, after syncing current sources:

```sh
PYTHONPATH=src python -m venturi.ops.build --component ops --architecture sm70 --output-dir build
PYTHONPATH=src VENTURI_OPS_LIBRARY=/absolute/path/to/libventuri_ops.so \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_q5_linear.py
PYTHONPATH=src python tools/benchmark_q5_adaptive.py \
  --library /absolute/path/to/libventuri_ops.so --output /absolute/path/to/results.json
```

The default benchmark uses three representative synthetic projection shapes, M=1..256 buckets, distinct
calibration/held-out random activations, alternating route order, and a shuffled
mixed-load trace. It records raw repeat samples, GPU-stream elapsed time, host wall
and process CPU time separately, selected routes, workspace/weight bytes, DSO and
weight hashes and sampled NVML telemetry. These are Linear-call timings, not pure
kernel attribution or whole-model tokens/s. `--checkpoint` uses matching original
Q5 planes from an existing safetensors artifact without requantizing; activations
are still synthetic. Use a separate `--profile-only` run for Nsight diagnostics;
never use profiler timings as unprofiled benchmark results.

No QPN-style Q5 kernel, arbitrary precision/format migration, online autotuning,
cross-GPU deployment, full-model Q5 quality promotion or production-default change
is implied. Those need their own integration and qualification campaigns.

## Measured 2026-11-17 campaign

WZU_Server GPU0: V100-PCIE-32GB, Torch 2.7.1+cu126, loaded cuBLAS 12.6.4,
NVCC 12.9.86. The existing Q5 artifact supplied real weight planes; activations
were synthetic and held out from calibration. No whole-model TPS is measured.

| Scratch cap | N5120/K6144 | N17408/K5120 | N5120/K17408 |
|---|---:|---:|---:|
| 64 MiB | 1.197x | 1.384x | 1.430x |
| 256 MiB | 1.144x | 1.372x | 1.139x |

These are mixed-load call-time ratios against same-budget reconstruction+cuBLAS,
not model generation speedups. A resident-FP16 cuBLAS control remains faster in
most cells. Only the gate-projection M1 cell has a clear >3% win over that control.
A 0.912x same-cuBLAS-route cell in the 64 MiB campaign is retained; a separate
handle/workspace cross-probe did not reproduce it, so per-cell non-regression is
not claimed. The 256 MiB campaign has a worst cell of about 0.969x.

See `operations/q5-adaptive-sm70-20261117/summary.json` and the delivered raw
evidence archive for identities, limits, repeated samples and service restoration.
