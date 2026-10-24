# Same-input projection groups

## Scope and invariants

This change is a scheduling/transport optimization of the existing SM70
`q8-residual` profile, not a new precision profile. It does not change checkpoint
weights, group-32 FP16 scales, either activation quantization pass, FP32 group
FMA/reduction order, FP16 output rounding, attention/GDN equations, KV precision,
request cursors, EOS or admission policy. Native token/layer loops remain C++.
Default FP16, SM89, streaming, PP and non-host TP retain the old implementation.
The public blocks ABI remains 16; native build dependencies now include nine
files because `q8_fused.cuh` is explicit in the build key.

`shared_linears` already avoids duplicate input quantization/copies for MLP
gate/up, GDN qkv/z and attention q/k/v. Previously each projection still
recorded/waited on both cross-device events, gathered its output, and reused
one peer output span. This serialized work that shares a producer and consumer.

The new grouped path gives each peer result a distinct admitted output span,
records one producer event for the input group, computes the group, copies the
combined peer output once, records one peer completion event, then gathers the
individual pitched slices on the primary. Both cross-device fences remain;
input, output and quantization scratch cannot be overwritten before their
previous consumers complete. Host transport remains explicit; P2P is not
assumed merely because CUDA advertises access. Transfer payload bytes do not
shrink. Scratch admission covers the sum of grouped shard rows, including tiny
fixtures; the tested Qwen vocabulary head already sets a larger bound.

## Source ablation

- Mode 0: pinned previous K-tiled library, original per-matrix fences/kernels.
- Mode 1: grouped transport, unchanged individual projection kernels.
- Mode 2: grouped transport plus one complete-dot GPU kernel per group/device.

The source-only compile-time choice `VENTURI_Q8_GROUP_MODE` is recorded in each
executed snapshot and library hash; it is not a new user-facing precision or
runtime toggle. Mode 2 passes two/three descriptors by value to the kernel.
Each CTA selects one matrix once from prefix row-block counts. A row is computed
entirely on one GPU; there are no partially rounded FP16 sums or inter-matrix
reductions. Output descriptors support full-width primary pitches and compact
peer spans. Mode 2 also eliminates the grouped primary scratch-to-output copy
by writing the full-width pitch directly; thus the mode-1/mode-2 serving
ablation is a fusion-plus-strided-store bundle, not pure kernel fusion alone.
Quantized input is explicitly packed once per group; no pointer-
identity cache survives a group, layer, forward, failure, request or reset.

T1 preserves the original global-dot body, eligible T2 uses full activation staging, and
larger T uses K-tiled staging. Fused large/small attention projections share the
large-grid geometry; this choice is tested rather than inferred from occupancy.
The unpromoted operator also supports explicit one-pass A8 research tests, but
native model dispatch still requires the already-qualified two-pass profile.

## Correctness and scope of reference reuse

Both mode 1 and mode 2 execute fresh trained candidate checks: 98 full-vocabulary
logit vectors, four native greedy trajectories, committed scalar private states,
eight heterogeneous batched slots, and selected-reset/continuation state. Every
signature matches the pinned previous residual candidate, not merely top-1 IDs.
The unchanged control signatures are hash-pinned and reused rather than loaded
again. The helper now allows an explicit affected-profile selection and records
unexecuted FP16 references; this is not a fresh trained FP16 qualification.

Both candidates also execute fresh four-case MTP output/committed-target-state
gates with natural acceptance/replay/EOS and no forced proposals. The unchanged
FP16 draft block's independent represented-weight CPU gates are reused, not
rerun. The original independent target tolerances remain top-1, relative
L2 <= .02, centered cosine >= .999 and KL <= .002. No broad original-BF16 corpus
quality claim is added. Fixture coverage includes independent FP64 model/logit
gates, ragged/inactive slots, failure/reset, EOS/limits, aliases and model lifetime.

79 new fused-operator cases cover both pass counts, odd K/row/token tails,
unequal matrices, large-plus-tiny groups, pitched sentinel columns, graph replay,
nonfinite/zero inputs, overlap/preflight and raw invalid descriptors. They match
old output and packed-workspace bytes and pass an independent FP64 projection
reference. Final native fixture suite: 1517 passed / 440 skipped. Local CPU-only
suite: 765 passed / 1192 skipped; skipped CUDA tests are not claimed as local
GPU passes.

## Measurement protocol

The microbenchmark uses three real TP-first groups at T=1/2/4/8/23/32, separate
control-unfused, candidate-unfused and candidate-fused arms, three repeats and
nine graph-contained group calls/wave. Crucially every unfused group also packs
its shared input exactly once. This is not serving TPS. Very short event windows
do not provide reliable per-wave NVML utilization; broad telemetry is preserved.

Serving controls and candidates use the same original-derived Q8 checkpoint,
FP16 KV, two V100s in physical order 1,0, explicit host TP, resident FP16 target,
32 projection rows, eight slots, capacity 256, identical 23 prompt IDs and 32
fixed greedy outputs. EOS is ignored only for the fixed-work speed case. A
25-ms admission window is included in HTTP wall. C1/2/8 and MTP1 screens precede
conditional three-repeat C1/2/4/8 and single-request MTP1 measurements. Cohort
TPS uses all emitted outputs/global cohort wall, never summed request durations.
Native post-first timings, TTFT, CPU core-seconds and client latencies are
separately labelled. Loading, warmup/reset and profilers are outside timed waves.

Separate fresh C1/C8 traces compare mode 0/1/2; a separate MTP1 trace and
standalone operator traces inspect dispatch, gaps, events/copies and compiled
register/shared-memory limits. Nsight runs disable CUDA event tracing to avoid
the previously observed instrumentation hang. Compiled residency ceilings and
NVML busy/controller percentages are not achieved occupancy or DRAM bandwidth.
Driver counter policy stays restricted and unchanged; unavailable counters are
reported as null rather than estimated.

External non-spec and successful MTP depth 1-4 controls remain hash-pinned in
previous immutable capsules, explicitly reused rather than rerun or selected
weaker. Native/GGUF Q8 encodings, scales/rounding, small parameters and internal
arithmetic are not identical. Failed external configurations remain failures,
not zero-speed controls. Native MTP2, batched MTP, paged KV, continuous batching,
sampling, whole-model graphs and broad/long-context quality remain unfinished.
