# Lossless small-parameter execution for Qwen Q8 / FP16

## Scope and outcome

The full goal remains **active**. This fixes the small-parameter rounding found
in the initial SM70/FP16 profile, without modifying the artifact or widening any
operator error criterion. A complete trained-model forward/generation/quality,
production native scheduling/sampling, whole-model graphs, paging/batching and
matched single/concurrent engine speed comparisons are still unfinished.

## Implementation

- Blocks **ABI 10** adds an explicit FP32-parameter / FP16-or-BF16 activation
  Linear and an independent parameter-dtype flag for causal convolution. Old
  ABIs fail before new-symbol binding. Exact SM70/SM89 architecture guards and
  the SM70 BF16 rejection remain. Ada Q4 is still SM89-only.
- `upload_fp32_linear(DenseWeight(fp32), dtype=..., device=...)` preserves the
  supplied FP32 tensor. Ordinary dense upload remains strict, not an implicit
  cast API. Native dot products use FP32 parameters/products/FMA/reduction and
  one activation-output cast, with no workspace, allocation or numerical Torch
  fallback in run(). This direct kernel is a fidelity baseline, not an optimized
  large GEMM claim. Mixed-weight Linear cannot accidentally reuse the old dense
  embedding ABI; it rejects lookup with a clear error.
- Convolution independently tracks activation dtype and weight dtype. Native
  FP32 taps are consumed before the observable conv-output activation cast and
  SiLU, with chronological raw history and safe exact history aliasing unchanged.
- The Qwen FP16 loader widens original BF16 a/b/conv values **losslessly to FP32**.
  Q8 codes/FP16 scales remain encoded and unchanged, activations/KV remain FP16,
  gamma/gates/recurrent state remain FP32. The BF16 profile and explicit FP64
  diagnostic mode retain their previous precision policy. No external model
  forward is invoked. All numerical native work is CUDA; orchestration remains
  the existing conservative synchronous Python baseline.
- Metadata-only memory planning includes these wider tensors, supports a
  selected layer, and is tested against actual resident CPU/native storage.
  These reusable operators are not tied to Qwen roles in their numerical kernels.

## Retained real counterexample and actual uploaded-value check

The original-derived artifact is unchanged:
`3d635b87db004f42aefd6730e372e530`, full file SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
The qualifier recomputed its complete hash and checked file stability.

The previous FP16 profile rounded layer-0 a/b/conv parameters:

| Parameter | Elements | Previously changed | Previously underflowed | Now changed / underflowed |
|---|---:|---:|---:|---:|
| a projection | 245,760 | 62 | 0 | **0 / 0** |
| b projection | 245,760 | 110 | 0 | **0 / 0** |
| convolution | 40,960 | 53 | 1 | **0 / 0** |

The convolution value at chronological `[tap=0, channel=7115]` was
`-1.7113052308559418e-08`, which previously became negative zero in FP16. The new
FP32 resident value matches the original exactly. The report compares **all
532,480 values in the checked real GDN layer**, not selected rows or dtype alone.
Maximum parameter error is zero; none became nonfinite. This is not a GPU upload
audit of all 48 GDN layers or a complete trained-model quality claim.
`metadata-and-counterexample.json` retains indices/values and the old casting
counterexample; `../sm70-fp16/` remains immutable historical evidence.

Tests additionally use tiny BF16 weights that vanish in FP16, amplify them with
finite FP16 inputs, and require nonzero independently computed output. This
checks that native compute—not merely storage—avoids premature weight casting.
An intentionally zeroed resident value must fail the fidelity checker.

## Final validation

| Environment | Scope | Result |
|---|---|---|
| Local Mac | Full suite; opt-in CUDA cases skipped, compiler orchestration mocked | **653 passed, 299 skipped** |
| 4090D, Torch 2.9.1+cu130 / nvcc 13.0.88, SM89 ABI 10 | Full suite, native FP16/BF16 regressions | **920 passed, 32 skipped** |
| V100, unchanged Torch 2.5.1+cu121 / nvcc 12.9.86, SM70 ABI 10 | Selected small-parameter/FP16-port/tool/build tests, CPU + native cases | **61 passed, 12 skipped** |

New checks cover mixed projection oracles/storage ownership/aliases and explicit
API rejection, FP32-tap convolution K=1/4 and chunked history, actual GDN parameter
binding, whole-layer byte planning, native FP16/BF16 graph capture with tensor
allocation/view/cast and numerical fallback prohibited. V100 BF16-native tests
are explicitly skipped, not emulated. Existing hybrid composition/chunking,
request/stream ownership, failure poisoning/reset and all earlier regressions
remain covered. No operator bound was relaxed; historical rounding/cancellation
counterexamples are not reclassified as fixed by this parameter correction.

The independent real-weight layer tool now uses **original** small parameters
for its CPU FP64 oracles, not FP16-rounded parameters. It checks each stage on its
actual native input. Q8 row tiles decode to FP64 without rounding represented
weights. GDN layer 0 and full-attention layer 3 at T=1 and T=8 pass **70 existing
per-operator checks**, with exact native manual-composition/state/cache parity
and actual-input fused/staged RoPE parity. These use synthetic activations, not
trained-model hidden states. The complete model is still unqualified.

The real-layer job finished normally in ~108 seconds including full file hash,
uploads, CPU oracles and synchronization: **not** layer latency, decode TPS,
CPU efficiency or a performance comparison.

## Build provenance

Both final builds hash CUDA source:
`75b87c140fa07feebabcc3ba353ce8b9798889d4ae6fda0faf7b8d3d01990aa4`.

- SM70: `308dc54fb9e2a32a/libventuri_blocks.so`, SHA256
  `94fa82adac3f2ea6d5c06f1405e15f342862f7e7fbca08cdc4ae83ed9c9de783`.
- SM89: `9bd89e952bc8b5a6/libventuri_blocks.so`, SHA256
  `4c14b35fc2ab5cf99125975163ade8a1c746fc8bf11f788fc1b391538eb0b2c8`.

`v100/fp32-small-qualification/` and `sm89/` retain build commands/output, hashes,
GPU state, tests and actual-layer reports. `source/` / `source-sha256.json` freeze
the current code/tests/docs. CUDA builds ran on remote Linux only; no Rust
compilation, local CUDA build, environment/package modification, commit or push
occurred. Existing external runner SHA256 is unchanged.

## Capacity, diagnostics and remaining work

Actual layer weights: **408,822,656 bytes** for GDN layer 0 and **395,552,768**
for attention layer 3. Full native FP16 weight tensors are planned at
**28,654,073,856 bytes**, 51,118,080 bytes above the unchanged BF16-profile weight
budget. Request state/KV, CUDA context/allocator, workspaces/logits/graphs are
additional. Exact storage comparisons test the planner; full model loading was
not attempted into insufficient VRAM. It still cannot fully reside on a 24 GiB
4090D or the currently free V100 memory.

Existing GPU services stayed running. GPU state was sampled for correctness
context only; ~3 GiB was free on the chosen V100 and ~9 GiB on the 4090D. There
is no new isolated baseline/change timing or throughput result. No speed,
CPU-efficiency, occupancy or bandwidth improvement is claimed. The known direct
Q8 prefill regression remains an open bottleneck. Hardware-counter policy
`RmProfilingAdminOnly=1` is unchanged: achieved occupancy, SM/DRAM/cache/stall
counters were not collected, and no repeated profiler campaign was run. NVML
snapshot busy percentage is not achieved occupancy or bandwidth utilization.

Next: execute/qualify the **complete trained model**. If a sufficiently free GPU
is unavailable, add a clearly separate explicit layer-streaming/offload
correctness mode rather than secretly changing residency or precision. Such a
mode must not replace the production-native single/concurrent serving target or
its matched-checkpoint/precision/context/offload-policy comparisons. Native
runtime/sampling/graphs, batch-aware recurrent state/paged KV and gap-driven
GEMM optimization remain required before the full goal can be complete.
