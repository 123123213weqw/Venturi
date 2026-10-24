# Bounded INT16-activation DP2A research prerequisite

The preceding native column-MLP profile improves actual single decode to47.629
TPS but needs another9.79% to meet the full external52.293 TPS floor. Previous
turn is progress, not a wait or a completed goal. Preserve that qualified native
path and all37 frozen capsules. No production dispatch change in this pilot.

## Candidate and declared arithmetic

One INT16 group32 activation pass, scale maxabs/32767 in FP32, nearest-even codes
clamped[-32767,32767], original stored Q8 weights/FP16 scales. Signed DP2A srcA
holds two INT16 values; lo/hi select two INT8 values from srcB. Four independent
integer chains, one FP32 group scale/FMA, original warp reduction. FP16 output
for full row projections; FP32 for actual column down partials. Invalid activation
groups become zero codes/NaN scales; tails are zero. This is new arithmetic, not
old FP16 or two-pass-residual byte equivalence; no whole-model quality claim.

Compile a separate SM70-only research SO on WZU_Server. No production blocks
dependency, model constructor/default or stored checkpoint changes. Full source
sync/hash assertion precedes compilation. No local CUDA/Rust compilation. Preserve
ABI/alignment/geometry/alias guards, independent integer/FP64 checks and failures.

## Finite operator experiment and gates

First: CPU reference/geometry plus remote signed-integer extremes/lane mapping,
exact quantizer codes/scales, tails/padding/strides, FP16 and FP32 sampled/full
FP64 checks, invalid/nonfinite/zero/large/tiny values, graph replay/changing input
and no Torch math during native calls. Only passing arithmetic proceeds to timing.

Then three alternating unprofiled trials per arm/cell, same preallocated original
checkpoint planes and normalized-state-shaped FP16 input. Actual first trained
GDN QKV/Z, attention Q/K/V and MLP gate/up row halves at T1/3/8, column-split MLP
down FP32 at T1/3/8, and GDN/attention output row halves at T1/3/8:18 cells. Tokens
are operator rows, not concurrent requests. Both arms include quantization and
complete projection; graphs reduce Python noise but are not a model graph claim.
Keep matched graph nodes/replays, event timing, CPU core-seconds and wall latency.

Promotion requires >=1.05 median fused-operator throughput ratio and **no tested
cell worse than0.98**, each based on three trials. No tile/register/row sweep.
Below gate: preserve negative evidence, do not launch a model/HTTP campaign or
change qualified model TPS. Above gate: only then integrate a **new named target
profile** and require fresh teacher/private-state/MTP checks, C1/C8 screens and
promoted three-repeat1/2/4/8 matrices. That later model work, not this prerequisite,
must prove the unchanged full external >=5% objective.

NVML100ms sample GPU busy/memory/power/temperature/clocks; separate timeline and
compiled-resource inspection. Profiled times never enter unprofiled rates.
Achieved occupancy/DRAM/cache/stall counters remain unavailable under unchanged
RmProfilingAdminOnly1; no repeated denied campaign or driver privilege edits.
Strict direct SSH/verified host keys, finite recovery, no unrelated job changes,
permanent services, automatic host switch, commit or push.
