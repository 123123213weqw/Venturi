# FP32 column/row MLP TP prerequisite: block gain, not model TPS

## Implemented

Explicit SM70 CUDA partial projection and FP32-sum/FP16-cast exports, Python
preparation with strict scratch/alias/device validation, and encoded group32 Q8
column slicing without requantization. No model dispatch uses these exports yet.
The old native TP/PP/MTP defaults and draft arithmetic remain unchanged.

Two V100-PCIE32GB, logical0/1=physical1/0, pinned host bridge. Reused verified
Qwen3.5-27B target artifact `3d635b87db004f42aefd6730e372e530`; official parent
revision `fc05daec18b0a78c049392ed2e771dde82bdf654`. Actual first-layer MLP
H5120/I17408 gate/up/down Q8 planes, FP16 scales, FP16 normalized inputs/output,
two-pass residual-A8 with FP32 scales. Both arms start/end **primary-only**.
No norm, residual, attention/GDN/cache, MTP, scheduler or complete model is timed.
T below is token rows per block call, **not concurrent HTTP request count**.

## Unprofiled matched result

Three alternating trials/arm/cell,400 block calls/trial, preallocated four-call
CUDA graphs, both cross-device ownership fences. Complete joined primary CUDA
events cover both ranks; host wall and process CPU measured separately.

| T | Old row TP ms/call | Column/row ms/call | Throughput ratio | CPU core-us/call old→new |
|---:|---:|---:|---:|---:|
|1|0.2979997|0.2405128|1.2390×|298.515→240.905|
|3|0.3678392|0.2858547|1.2868×|368.485→286.227|
|8|0.5967388|0.4535329|1.3158×|713.508→548.673|

All three paired trials improve in every cell. Latency decreases19.3–24.0%; do
not confuse a24–32% throughput increase with that latency percentage. Whole-model
TPS, HTTP cohort latency and TTFT are not measured by this operator experiment.
There is no new win against llama.cpp or automatic serving-profile promotion.

The initial replicated-column-input/output variant is retained as `mlp-v3`;
the table uses corrected matched primary-only contract `mlp-v4`. Its first timed
wave starts after the previous profiler/worker completed: no overlap contaminates
these results. One unnecessary recovery worker expected a failed predecessor;
its assertion stops it after the original fixture worker succeeds. It runs no
duplicate GPU tests; preserve its failure rather than inventing a compiler fault.

## Arithmetic and correctness limits

Library SHA256 `d7001c142b942af3fe516bc5f00eb828ad2ce24e2716186c9b562301081fb9df`;
ten native build dependencies. Focused32 tests pass; full remote2118 pass/687
skip, local final800 pass/2007 skip. Existing original-source/ABI/TP/PP/MTP fixture
coverage runs in that suite; this is not fresh complete-model quality approval.

Local gate/up/SwiGLU halves equal the previous full intermediates byte-for-byte.
Down splits preserve original group scales and activation quantization bytes;
partials and transport are FP32, summed before one FP16 cast. Independent
represented FP64 projection checks pass. New/old FP16 outputs differ at8/19/44
positions (T1/3/8), relative L2≤1.256e-6, so **not same-profile byte parity**.
Tail/padding ownership, quantizer packing, FP32 output error, cancellation/large
partials, nonfinite/zero, aliases, graph replay and allocation-free native
execution are tested. No broad BF16 corpus or full new-profile state claim.

## Separate diagnostics

Profile node attribution confirms84992*T→30720*T actual H2D bytes and the same
D2H counts per call:63.86% fewer payload bytes. Transfer calls decrease from
5H2D+4D2H to2+2; kernels increase9→11 due to per-rank SwiGLU and the FP32 sum.
At T3/12 instrumented calls, summed kernel durations5.321→5.529ms increase,
while joined host span4.776→3.827ms decreases: reduced serialization/transport
and overlap matter, not merely lower summed kernel work. Process activity gaps
are not device-wide idle time, and sums are not critical-path shares.

NVML100ms samples are sparse (3–7/arm/device/cell) and lagging. T8 GPU busy means
70.3/68.3% old versus78.4/77.0% new, clocks/power/temperature and all per-wave
samples preserved in metrics. Both arms' weights reside together; device-used
3785/4247MiB at T1/3 and3787/4249MiB at T8 include unrelated baseline allocations.
These are not full-model VRAM requirements. Post-exit baseline3651/3189MiB restored.

Compiled partial kernels use48/44/50 registers,128/256/512 threads and0/9216/18432
shared bytes (T1/4/8), no local-memory allocation. Driver residency ceilings
62.5/62.5/50% are theoretical, not achieved occupancy. Counter policy remains
`RmProfilingAdminOnly:1`; achieved occupancy/DRAM/cache/stall metrics unavailable.
No privilege changes, persistent jobs, sampler, service, commit or push remains.

## Next

Integrate an opt-in **MLP-only** column/row native model path, not an immediate
whole-attention/GDN rewrite. Replace down weight storage instead of duplicating
it, admit FP32 scratch, preserve both TP fences and original private state.
Fresh model/teacher/state/MTP gates precede C1/C8 screens and promoted repeat
matrices. The detailed implementation contract is retained alongside this file.
The >=5% external single-decode and1/2/4/8-request goal remains active/incomplete.
