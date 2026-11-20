# Semantic head-local GDN pilot protocol

Research-only native host orchestrator, reusing the qualified production CUDA
blocks library. No production model/session/batch/MTP dispatch is changed. The
accepted engine remains the default; this pilot does not measure model TPS.

## Fixed comparison and gates (before trained timings)

Use the official Qwen3.5-27B Q8 artifact and original BF16 source at the verified
revision already pinned by the previous capsule. Test actual GDN layers 0 and
62 on both V100s, with explicit host transport and primary-only FP16 block
input/output. Row TP includes input RMSNorm, QKV/Z and FP32 A/B projections,
primary convolution/gates/GDN, gated norm, row-split output and residual add.
Head TP includes the same operations but semantically splits corresponding
Q/K/V heads, A/B, convolution, private packed states and histories across ranks;
uses FP32 column partials/transport/sum followed by one FP16 output cast. This
last output projection changes FP32 association; encoded weights, scales and
two-pass residual-A8 arithmetic are otherwise retained. It is not a byte-exact
original-output or BF16 quality claim.

Compare joined pre-output arrays, states, histories and prefix1/2 byte-for-byte.
Output relative L2 versus the row arm must be below 5e-4. Independent
original-source represented-Q8/FP64 block reference retains relative L2 <=0.02
and cosine >=0.999; do not loosen tolerances to obtain a pass. Synthetic
fixtures additionally test private eight-request selective reset, repeated
native buffer reuse, MTP prefix restoration followed by shadow continuation,
and prohibit Python numerical operations during native runs. An ineligible
odd-width original fixture stays unchanged; the pilot has a separate eligible
fixture rather than relaxing column alignment.

Workloads: one sequence/one token; one sequence/three verification tokens with
two prefix backups; eight independent sequences/one token each. No substitution
of a single eight-token recurrence. Three alternating native CPU-dispatch-
inclusive unprofiled trials per arm/cell, 8 warmup blocks and 2000 measured
blocks, equal recurrence count and final state checks. Record native wall,
process CPU core-seconds, transfer bytes/calls, pinned memory and sampled GPU
busy, memory-controller busy, VRAM, power, temperature and clocks. Both operator
arms are resident, so their VRAM is not a model-serving admission measurement.

Promote only if both trained three-token cells achieve >=1.10x speedup and no
one-token or independent-eight cell regresses >2%. A positive operator result
only permits later model ownership integration; that integration still needs
trained teacher/MTP/state tests and C1/C8 screens before a full cohort matrix.
Negative results stop before the model ownership rewrite.

Collect an independent, shorter Nsight Systems trace (32 blocks per arm,
layer0 only), with NVTX/API/correlation ownership, actual launch geometry,
kernel/copy attribution, process activity gaps and two-device overlap. Profiled
timings are not throughput results. Hardware counters remain unavailable under
the existing driver policy; no privilege or policy changes and no repeated
counter replay campaign. Static register/shared usage is not achieved occupancy.
Preserve every failed build/test attempt and its executed source.

## Dispatch-fidelity correction after the first trained pilot

Initial fixtures-v3/trained-v5/profile-v6 used one gate kernel per independent
sequence in both arms. Inspection of the production batch dispatcher shows its
A/B gates are batched once per rank, with only convolution and GDN iterated per
request. Preserve those initial results, but do not use them as the final
integration gate. Source-v4 corrects this dispatch mismatch for both arms;
repeat fixtures, trained timings and the independent trace. Scalar T1/T3
dispatch and all numerical gates are unchanged. This is not a precision or
tolerance change.
